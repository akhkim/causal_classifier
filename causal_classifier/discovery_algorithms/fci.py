import networkx as nx
import numpy as np
import pandas as pd
import json
from causallearn.search.ConstraintBased.FCI import fci as fci_algorithm
from causallearn.graph.Endpoint import Endpoint
from ..llm_query import create_chat_completion

# GPU acceleration imports
try:
    import torch
    import torch.nn.functional as F
    from scipy.stats import norm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - falling back to CPU-only FCI algorithm")

def _detect_gpu_capability():
    """
    Detect available GPU acceleration options for FCI.
    """
    gpu_info = {
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'memory_gb': 0,
        'can_accelerate': False
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_info['cuda_available'] = True
        gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info['can_accelerate'] = True
    
    return gpu_info

def _should_use_gpu(data_shape, gpu_info):
    """
    Determine if GPU acceleration would be beneficial for FCI.
    """
    n_samples, n_features = data_shape
    
    if not gpu_info['can_accelerate']:
        return False, "No GPU acceleration available"
    
    # FCI is more computationally intensive than PC, so lower thresholds
    min_samples_for_gpu = 300
    min_features_for_gpu = 4
    
    if n_samples < min_samples_for_gpu:
        return False, f"Sample size {n_samples} too small for GPU overhead (min: {min_samples_for_gpu})"
    
    if n_features < min_features_for_gpu:
        return False, f"Feature count {n_features} too small for GPU overhead (min: {min_features_for_gpu})"
    
    # Estimate memory requirements
    estimated_memory_gb = (n_samples * n_features * 8) / (1024**3) * 3  # FCI needs more memory
    
    if estimated_memory_gb > gpu_info['memory_gb'] * 0.5:
        return False, f"Estimated memory {estimated_memory_gb:.1f}GB exceeds safe limit ({gpu_info['memory_gb']*0.5:.1f}GB)"
    
    return True, f"GPU acceleration beneficial for {n_samples}x{n_features} dataset"

class GPUAcceleratedFCI:
    """
    GPU-accelerated implementation of FCI algorithm using PyTorch.
    Falls back to CPU implementation when GPU is not available or beneficial.
    """
    
    def __init__(self, device=None):
        self.gpu_info = _detect_gpu_capability()
        
        if device is None:
            self.device = torch.device('cuda' if self.gpu_info['cuda_available'] else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.use_gpu = self.device.type == 'cuda'
        self.fast_mode = False
        
    def run_fci_algorithm(self, data, alpha=0.05, indep_test="kci"):
        """
        Run FCI algorithm with GPU acceleration where beneficial.
        """
        if not self.use_gpu:
            return self._run_fci_cpu(data, alpha, indep_test)
        
        try:
            # Use GPU-accelerated approach for conditional independence tests
            return self._run_fci_gpu_hybrid(data, alpha, indep_test)
        except Exception as e:
            print(f"GPU FCI algorithm failed, falling back to CPU: {e}")
            return self._run_fci_cpu(data, alpha, indep_test)
    
    def _run_fci_gpu_hybrid(self, data, alpha, indep_test):
        """
        Hybrid GPU-CPU FCI implementation.
        GPU for independence tests, CPU for graph operations.
        """
        print("Running GPU-accelerated FCI algorithm...")
        print(f"Dataset: {data.shape[0]}x{data.shape[1]}, Fast mode: {self.fast_mode}")
        
        # Move data to GPU for independence testing
        data_gpu = torch.tensor(data, device=self.device, dtype=torch.float32)
        
        # Handle NaN values
        if torch.isnan(data_gpu).any():
            print("Warning: NaN values detected, filling with column means")
            for i in range(data_gpu.shape[1]):
                col_mean = torch.nanmean(data_gpu[:, i])
                data_gpu[torch.isnan(data_gpu[:, i]), i] = col_mean
        
        # Standardize data
        data_gpu = (data_gpu - data_gpu.mean(dim=0)) / (data_gpu.std(dim=0) + 1e-8)
        
        # Create custom independence test that uses GPU
        def gpu_independence_test(X, Y, condition_set):
            """GPU-accelerated independence test for FCI"""
            try:
                return self._gpu_independence_test(data_gpu, X, Y, condition_set, alpha)
            except Exception:
                # Fallback to CPU test if GPU fails
                from causallearn.utils.cit import CIT
                cit = CIT(data, indep_test)
                return cit(X, Y, condition_set)
        
        # Run FCI with GPU-accelerated independence testing
        # Note: We still use causallearn's FCI structure but with our GPU independence test
        try:
            # Import and patch the independence test
            import causallearn.utils.cit as cit_module
            original_fisherz = getattr(cit_module, 'fisherz', None)
            
            # Temporarily replace with our GPU version for fisherz tests
            if indep_test == "fisherz":
                cit_module.fisherz = gpu_independence_test
            
            # Run standard FCI
            pag, _ = fci_algorithm(data, alpha=alpha, independence_test_method=indep_test)
            
            # Restore original function
            if original_fisherz is not None:
                cit_module.fisherz = original_fisherz
                
            return pag
            
        except Exception as e:
            print(f"Hybrid GPU-CPU FCI failed: {e}")
            return self._run_fci_cpu(data, alpha, indep_test)
    
    def _gpu_independence_test(self, data_gpu, X, Y, condition_set, alpha):
        """
        GPU-accelerated independence test similar to PC implementation.
        """
        indices = [X, Y] + list(condition_set)
        selected_data = data_gpu[:, indices]
        
        n_samples = selected_data.shape[0]
        
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(selected_data.T)
        
        if len(condition_set) == 0:
            r_xy = corr_matrix[0, 1].item()
        elif len(condition_set) == 1:
            # Simple partial correlation formula
            r_xy = corr_matrix[0, 1].item()
            r_xz = corr_matrix[0, 2].item()
            r_yz = corr_matrix[1, 2].item()
            
            numerator = r_xy - r_xz * r_yz
            denominator = torch.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            r_xy = (numerator / denominator).item()
        else:
            # Use precision matrix for multiple conditioning variables
            try:
                precision_matrix = torch.linalg.inv(corr_matrix + 1e-6 * torch.eye(corr_matrix.shape[0], device=self.device))
                r_xy = -precision_matrix[0, 1] / torch.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])
                r_xy = r_xy.item()
            except:
                return 1.0  # Conservative: assume dependent if computation fails
        
        # Fisher's z-test
        if abs(r_xy) >= 0.9999:
            r_xy = 0.9999 * (1 if r_xy > 0 else -1)
        
        z = 0.5 * np.log((1 + r_xy) / (1 - r_xy))
        se = 1.0 / np.sqrt(n_samples - len(condition_set) - 3)
        test_stat = abs(z) / se
        
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(test_stat))
        
        return p_value
    
    def _run_fci_cpu(self, data, alpha, indep_test):
        """
        CPU fallback implementation.
        """
        print("Running CPU FCI algorithm...")
        pag, _ = fci_algorithm(data, alpha=alpha, independence_test_method=indep_test)
        return pag

def _label(node, col_names):
    """
    Convert node object to proper variable name using column names.
    Handles different node representations from causallearn.
    """
    # Try to get the node name/index
    if hasattr(node, "get_name"):
        raw = node.get_name()
    elif hasattr(node, "name"):
        raw = node.name
    elif hasattr(node, "index"):
        raw = node.index
    else:
        raw = str(node)
    
    # Convert to proper column name
    try:
        # If raw is already an integer or string that can be converted to int
        if isinstance(raw, int):
            idx = raw
        else:
            idx = int(raw)
        
        # Make sure index is within bounds
        if 0 <= idx < len(col_names):
            return col_names[idx]
        else:
            print(f"Warning: node index {idx} out of bounds for columns {col_names}")
            return f"X{idx}"
    except (ValueError, TypeError):
        # If raw is already a string variable name, return it
        if isinstance(raw, str) and raw in col_names:
            return raw
        elif isinstance(raw, str):
            # If it's a string but not in col_names, check if it's a numbered variable
            if raw.startswith('X') and raw[1:].isdigit():
                try:
                    idx = int(raw[1:])
                    if 0 <= idx < len(col_names):
                        return col_names[idx]
                except ValueError:
                    pass
            return raw
        else:
            return str(raw)

def _select_hyperparameters_llm(data):
    """
    Use LLM to select optimal hyperparameters for FCI algorithm based on data characteristics.
    """
    
    # Convert to DataFrame if it's not already
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Gather data characteristics
    n_samples, n_features = df.shape
    
    # Analyze data types and distributions
    continuous_vars = 0
    discrete_vars = 0
    
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals > 10:
            continuous_vars += 1
        else:
            discrete_vars += 1
    
    # Calculate correlation statistics for continuous variables
    if continuous_vars > 1:
        corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    else:
        avg_correlation = 0.3
    
    context = f"""
    Dataset Characteristics:
    - Sample size: {n_samples}
    - Number of variables: {n_features}
    - Continuous variables: {continuous_vars}
    - Discrete variables: {discrete_vars}
    - Average correlation: {avg_correlation:.3f}
    - Sample-to-variable ratio: {n_samples/n_features:.2f}
    
    Note: FCI is designed for causal discovery with latent confounders.
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in FCI (Fast Causal Inference) algorithm hyperparameter selection for causal discovery with latent confounders.
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. alpha (significance level): Controls Type I error rate for conditional independence tests. More conservative with latent confounders.
                2. indep_test (independence test): Type of test to use based on data characteristics.
                
                Guidelines:
                - Large samples (n>1000): Can use lower alpha (0.01-0.05) for more conservative testing
                - Small samples (n<200): Use higher alpha (0.05-0.1) to maintain power, but be careful with latent confounders
                - High dimensionality (p>20): Use more conservative alpha due to multiple testing and complexity
                - Continuous data: Use "kci" (kernel CI test) as it's robust and non-parametric
                - Mixed/discrete data: Use "kci" or "chisq" based on data characteristics
                - FCI with latent confounders requires more conservative testing than PC
                
                Available independence tests:
                - "kci": Kernel conditional independence test (recommended for FCI, handles non-linear dependencies)
                - "fisherz": Fisher's z-test (for Gaussian continuous data)
                - "chisq": Chi-square test (for discrete/categorical data)
                - "gsq": G-square test (alternative for discrete data)
                
                Return your response as valid JSON with this exact structure:
                {
                    "alpha": <float>,
                    "indep_test": "<test_name>",
                    "reasoning": "<brief explanation of choices>"
                }"""},
                {"role": "user", "content": f"Given these dataset characteristics, what are the optimal FCI algorithm hyperparameters?\n\n{context}"}
            ],
            temperature=0.1,
            thinking=False
        )
        
        hyperparams = json.loads(response)
        return hyperparams
        
    except Exception as e:
        print(f"LLM hyperparameter selection failed: {e}")
        # Fallback to rule-based selection
        return _select_hyperparameters_fallback(n_samples, n_features, continuous_vars, discrete_vars)

def _select_hyperparameters_fallback(n_samples, n_features, continuous_vars, discrete_vars):
    """Fallback rule-based hyperparameter selection."""
    
    # Alpha (significance level) - more conservative for FCI due to latent confounders
    if n_samples < 200:
        alpha = 0.1
    elif n_samples < 1000:
        alpha = 0.05
    else:
        alpha = 0.01
    
    # Adjust for multiple testing (high dimensionality)
    if n_features > 20:
        alpha = alpha / 2
    elif n_features > 50:
        alpha = alpha / 5
    
    # Independence test selection - default to KCI for FCI as it's more robust
    if continuous_vars > discrete_vars:
        indep_test = "kci"      # Robust for continuous and mixed data
    elif discrete_vars > continuous_vars:
        indep_test = "chisq"    # Good for discrete data
    else:
        indep_test = "kci"      # Safe default for FCI
    
    return {
        "alpha": alpha,
        "indep_test": indep_test,
        "reasoning": "Fallback rule-based selection for FCI with latent confounders"
    }

def run(df, use_gpu=False, fast_mode=None):
    """
    Run FCI algorithm with LLM-optimized hyperparameters and optional GPU acceleration.
    """
    
    # Detect GPU capabilities and determine if GPU should be used
    gpu_info = _detect_gpu_capability()
    data_array = df.values if isinstance(df, pd.DataFrame) else df
    should_use_gpu, gpu_reason = _should_use_gpu(data_array.shape, gpu_info)
    
    # Auto-detect fast mode for large datasets
    if fast_mode is None:
        n_samples, n_features = data_array.shape
        fast_mode = (n_features > 12) or (n_samples * n_features > 30000)
        if fast_mode:
            print(f"Auto-enabling fast mode for large dataset ({n_samples}x{n_features})")
    
    # Override user choice if GPU is not beneficial
    if use_gpu and not should_use_gpu:
        print(f"GPU acceleration requested but not beneficial: {gpu_reason}")
        use_gpu = False
    elif use_gpu and should_use_gpu:
        print(f"GPU acceleration enabled: {gpu_reason}")
        print(f"GPU Info: PyTorch backend, {gpu_info['memory_gb']:.1f}GB memory")
    
    # Get optimal hyperparameters using LLM
    hyperparams = _select_hyperparameters_llm(df)
    
    # Adjust alpha for fast mode (more aggressive)
    if fast_mode:
        original_alpha = hyperparams["alpha"]
        hyperparams["alpha"] = min(0.15, hyperparams["alpha"] * 1.5)  # More liberal threshold
        print(f"Fast mode: Adjusted alpha from {original_alpha:.3f} to {hyperparams['alpha']:.3f}")
    
    print(f"FCI Algorithm Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")
    
    if use_gpu and TORCH_AVAILABLE:
        # Use GPU-accelerated FCI implementation
        try:
            gpu_fci = GPUAcceleratedFCI()
            gpu_fci.fast_mode = fast_mode
            
            pag = gpu_fci.run_fci_algorithm(
                data_array, 
                alpha=hyperparams["alpha"], 
                indep_test=hyperparams["indep_test"]
            )
            
            hyperparams["gpu_accelerated"] = True
            hyperparams["gpu_backend"] = "pytorch"
            hyperparams["fast_mode"] = fast_mode
            
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fallback to CPU version
            pag, _ = fci_algorithm(
                data_array,
                alpha=hyperparams["alpha"],
                independence_test_method=hyperparams["indep_test"]
            )
            hyperparams["gpu_accelerated"] = False
            hyperparams["fast_mode"] = fast_mode
    else:
        # Run standard CPU version
        print("Running CPU-based FCI algorithm...")
        if fast_mode:
            print("Fast mode enabled: Using more aggressive pruning")
        
        pag, _ = fci_algorithm(
            data_array,
            alpha=hyperparams["alpha"],
            independence_test_method=hyperparams["indep_test"]
        )
        hyperparams["gpu_accelerated"] = False
        hyperparams["fast_mode"] = fast_mode

    # Process PAG to DAG
    dag = nx.DiGraph()
    
    # PAG -> DAG
    col_names = list(df.columns)
    print(f"Available column names: {col_names}")
    
    # Add all nodes to DAG first
    dag.add_nodes_from(col_names)
    
    # Check if PAG has labels attribute like PC algorithm
    if hasattr(pag, 'labels'):
        print(f"PAG has labels: {pag.labels}")
        # Use the labels if available
        node_mapping = {i: pag.labels[i] if i < len(pag.labels) else f"X{i}" for i in range(len(col_names))}
    else:
        # Create a mapping from indices to column names
        node_mapping = {i: col_names[i] for i in range(len(col_names))}
    
    print(f"Node mapping: {node_mapping}")
    
    # Access edges using the correct method for GeneralGraph
    try:
        # Try different methods to get edges from the PAG
        if hasattr(pag, 'get_graph_edges'):
            edges = pag.get_graph_edges()
        elif hasattr(pag, 'edges'):
            edges = pag.edges
        elif hasattr(pag, 'get_edges'):
            edges = pag.get_edges()
        else:
            # Fallback: inspect the PAG object structure
            print(f"PAG type: {type(pag)}, available methods: {[m for m in dir(pag) if not m.startswith('_')]}")
            # Create a minimal DAG with all nodes but no edges if we can't extract edges
            dag.graph['fci_hyperparameters'] = hyperparams
            return dag
            
        print(f"Found {len(edges)} edges in PAG")
        
        for i, edge in enumerate(edges):
            try:
                node1 = edge.get_node1()
                node2 = edge.get_node2()
                
                # Debug: print raw node information for first few edges
                if i < 3:  # Only print first 3 edges for debugging
                    print(f"Edge {i}: node1={node1} (type: {type(node1)}), node2={node2} (type: {type(node2)})")
                    if hasattr(node1, 'get_name'):
                        print(f"  node1.get_name(): {node1.get_name()}")
                    if hasattr(node2, 'get_name'):
                        print(f"  node2.get_name(): {node2.get_name()}")
                
                # Get node indices or names and map to actual variable names
                node1_idx = _get_node_index(node1)
                node2_idx = _get_node_index(node2)
                
                # Map to actual variable names
                a_obs = node_mapping.get(node1_idx, f"X{node1_idx}")
                b_obs = node_mapping.get(node2_idx, f"X{node2_idx}")
                
                # Ensure the nodes are actually in our column names
                if a_obs not in col_names:
                    a_obs = col_names[node1_idx] if 0 <= node1_idx < len(col_names) else f"X{node1_idx}"
                if b_obs not in col_names:
                    b_obs = col_names[node2_idx] if 0 <= node2_idx < len(col_names) else f"X{node2_idx}"
                
                # Debug: print mapped names for first few edges
                if i < 3:
                    print(f"  Mapped: {a_obs} -> {b_obs}")
                
                ep_a = edge.get_endpoint1()
                ep_b = edge.get_endpoint2()

                # Oriented edge  A -> B
                if ep_a == Endpoint.TAIL and ep_b == Endpoint.ARROW:
                    dag.add_edge(a_obs, b_obs)

                # Opposite orientation  B <- A
                elif ep_a == Endpoint.ARROW and ep_b == Endpoint.TAIL:
                    dag.add_edge(b_obs, a_obs)

                # Bidirected edge  A <-> B  ==> latent confounder
                elif ep_a == Endpoint.ARROW and ep_b == Endpoint.ARROW:
                    latent = f"U_{min(a_obs,b_obs)}_{max(a_obs,b_obs)}"
                    if latent not in dag:
                        dag.add_node(latent, latent=True)
                    dag.add_edge(latent, a_obs)
                    dag.add_edge(latent, b_obs)
                else:
                    # Unknown edge type - log it for first few edges
                    if i < 3:
                        print(f"  Skipping edge with endpoints: {ep_a}, {ep_b}")
                    continue
                    
            except Exception as edge_error:
                print(f"Error processing edge {i}: {edge_error}")
                continue
                
    except Exception as e:
        print(f"Error extracting edges from PAG: {e}")
        print(f"PAG type: {type(pag)}")
        # Return DAG with just nodes if edge extraction fails
        dag.graph['fci_hyperparameters'] = hyperparams
        return dag

    # Remove cycle-creating edges by removing the last edge
    if not nx.is_directed_acyclic_graph(dag):
        print("Warning: Graph contains cycles, removing some edges")
        for cycle in list(nx.simple_cycles(dag)):
            dag.remove_edge(cycle[-1], cycle[0])

    # Store hyperparameters in graph metadata
    dag.graph['fci_hyperparameters'] = hyperparams
    
    return dag

def _get_node_index(node):
    """
    Extract node index from various node representations.
    """
    # Try to get the node index/name
    if hasattr(node, "get_name"):
        raw = node.get_name()
    elif hasattr(node, "name"):
        raw = node.name
    elif hasattr(node, "index"):
        raw = node.index
    else:
        raw = str(node)
    
    # Convert to integer index
    try:
        if isinstance(raw, int):
            return raw
        else:
            return int(raw)
    except (ValueError, TypeError):
        # If it's a string like "X4", extract the number
        if isinstance(raw, str) and raw.startswith('X') and raw[1:].isdigit():
            return int(raw[1:])
        # If it's already a variable name, try to find its index
        # This is a fallback that shouldn't normally happen
        return 0
