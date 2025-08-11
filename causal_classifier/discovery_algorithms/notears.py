from notears.linear import notears_linear
import networkx as nx
import numpy as np
import pandas as pd
import json
from ..llm_query import create_chat_completion

# GPU acceleration imports
try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - falling back to CPU-only NOTEARS algorithm")

def _detect_gpu_capability():
    """
    Detect available GPU acceleration options for NOTEARS.
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
    Determine if GPU acceleration would be beneficial for NOTEARS.
    """
    n_samples, n_features = data_shape
    
    if not gpu_info['can_accelerate']:
        return False, "No GPU acceleration available"
    
    # NOTEARS benefits significantly from GPU for optimization
    min_samples_for_gpu = 100
    min_features_for_gpu = 5
    
    if n_samples < min_samples_for_gpu:
        return False, f"Sample size {n_samples} too small for GPU overhead (min: {min_samples_for_gpu})"
    
    if n_features < min_features_for_gpu:
        return False, f"Feature count {n_features} too small for GPU overhead (min: {min_features_for_gpu})"
    
    # Estimate memory requirements
    estimated_memory_gb = (n_samples * n_features * 8) / (1024**3) * 2
    
    if estimated_memory_gb > gpu_info['memory_gb'] * 0.7:
        return False, f"Estimated memory {estimated_memory_gb:.1f}GB exceeds safe limit ({gpu_info['memory_gb']*0.7:.1f}GB)"
    
    return True, f"GPU acceleration beneficial for {n_samples}x{n_features} dataset"

class GPUAcceleratedNOTEARS:
    """
    GPU-accelerated implementation of NOTEARS algorithm using PyTorch.
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
        
    def run_notears_algorithm(self, data, lambda1=0.1, max_iter=100):
        """
        Run NOTEARS algorithm with GPU acceleration.
        """
        if not self.use_gpu:
            return self._run_notears_cpu(data, lambda1, max_iter)
        
        try:
            # Use GPU-accelerated approach
            return self._run_notears_gpu(data, lambda1, max_iter)
        except Exception as e:
            print(f"GPU NOTEARS algorithm failed, falling back to CPU: {e}")
            return self._run_notears_cpu(data, lambda1, max_iter)
    
    def _run_notears_gpu(self, data, lambda1, max_iter):
        """
        GPU-accelerated NOTEARS implementation.
        """
        print("Running GPU-accelerated NOTEARS algorithm...")
        print(f"Dataset: {data.shape[0]}x{data.shape[1]}, Fast mode: {self.fast_mode}")
        
        # Move data to GPU
        X = torch.tensor(data, device=self.device, dtype=torch.float32)
        
        # Handle NaN values
        if torch.isnan(X).any():
            print("Warning: NaN values detected, filling with column means")
            for i in range(X.shape[1]):
                col_mean = torch.nanmean(X[:, i])
                X[torch.isnan(X[:, i]), i] = col_mean
        
        # Standardize data
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
        
        n, d = X.shape
        
        # Initialize weight matrix W on GPU
        W = torch.zeros(d, d, device=self.device, requires_grad=True)
        
        # Optimizer
        optimizer = optim.Adam([W], lr=0.001 if not self.fast_mode else 0.002)
        
        # NOTEARS optimization with GPU acceleration
        rho, alpha = 1.0, 0.0
        for iter_count in range(max_iter):
            optimizer.zero_grad()
            
            # Compute loss and constraint on GPU
            loss, h = self._compute_loss_and_constraint_gpu(X, W, lambda1, alpha, rho)
            
            loss.backward()
            optimizer.step()
            
            # Project to remove self-loops
            with torch.no_grad():
                W.diagonal().fill_(0)
            
            # Update lagrangian parameters
            if iter_count % 10 == 0:
                h_val = h.item()
                if h_val <= 1e-8:
                    break
                alpha = alpha + rho * h_val
                if h_val > 0.25:
                    rho *= 10
            
            # Progress reporting
            if iter_count % 20 == 0 and iter_count > 0:
                print(f"Iteration {iter_count}: loss={loss.item():.4f}, constraint={h.item():.6f}")
        
        # Convert back to CPU
        W_final = W.detach().cpu().numpy()
        
        print(f"GPU NOTEARS completed in {iter_count+1} iterations")
        
        return W_final
    
    def _compute_loss_and_constraint_gpu(self, X, W, lambda1, alpha, rho):
        """
        Compute NOTEARS loss and DAG constraint on GPU.
        """
        n, d = X.shape
        
        # Compute squared loss
        M = X @ W  # n x d
        R = X - M  # n x d
        loss = 0.5 / n * torch.sum(R ** 2)
        
        # Add L1 regularization
        loss = loss + lambda1 * torch.sum(torch.abs(W))
        
        # Compute DAG constraint h(W) using matrix exponential trace
        # h(W) = tr(exp(W ⊙ W)) - d
        W_squared = W * W
        if d <= 20:  # Use exact matrix exponential for small graphs
            h = torch.trace(torch.matrix_exp(W_squared)) - d
        else:  # Use polynomial approximation for larger graphs
            # Taylor series approximation: exp(A) ≈ I + A + A²/2! + A³/3! + ...
            h_approx = d  # trace(I)
            A_power = W_squared
            factorial = 1.0
            
            for k in range(1, 6):  # Up to 5th order
                factorial *= k
                h_approx = h_approx + torch.trace(A_power) / factorial
                if k < 5:
                    A_power = A_power @ W_squared
            
            h = h_approx - d
        
        # Add augmented Lagrangian terms
        loss = loss + alpha * h + 0.5 * rho * h * h
        
        return loss, h
    
    def _run_notears_cpu(self, data, lambda1, max_iter):
        """
        CPU fallback implementation using original NOTEARS.
        """
        print("Running CPU NOTEARS algorithm...")
        W = notears_linear(data, lambda1=lambda1, max_iter=max_iter, loss_type='l2')
        return W

def _select_hyperparameters_llm(df):
    """
    Use LLM to select optimal hyperparameters for NOTEARS based on data characteristics.
    """
    
    # Gather data characteristics
    n_samples, n_features = df.shape
    
    # Calculate correlation statistics
    corr_matrix = df.corr().abs()
    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    max_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
    
    # Calculate data scale and variance
    data_std = df.std().mean()
    data_range = (df.max() - df.min()).mean()
    
    # Check for sparsity patterns
    sparsity_indicator = (corr_matrix > 0.1).sum().sum() / (n_features * (n_features - 1))
    
    context = f"""
    Dataset Characteristics:
    - Sample size: {n_samples}
    - Number of variables: {n_features}
    - Average correlation: {avg_correlation:.3f}
    - Max correlation: {max_correlation:.3f}
    - Average std deviation: {data_std:.3f}
    - Average data range: {data_range:.3f}
    - Sparsity indicator: {sparsity_indicator:.3f}
    - Sample-to-variable ratio: {n_samples/n_features:.2f}
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in NOTEARS (continuous optimization for DAG learning) hyperparameter selection.
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. lambda1 (L1 regularization): Controls sparsity of the learned DAG. Higher values lead to sparser graphs.
                2. max_iter (maximum iterations): Number of optimization iterations. More complex data may need more iterations.
                3. w_threshold (weight threshold): Edges with absolute weight below this are removed. Controls final graph sparsity.
                
                Guidelines:
                - Small samples (n<100): Use higher regularization (lambda1: 0.1-0.3), lower threshold (0.2-0.3)
                - Large samples (>500): Can use lower regularization (lambda1: 0.01-0.05), higher threshold (0.3-0.5)
                - High correlation (>0.5): May need more iterations (150-300), moderate regularization
                - Low correlation (<0.2): Use higher regularization to avoid spurious edges
                - High dimensionality (p>10): Use higher regularization and threshold
                - Small p/n ratio (<0.1): Can use lower regularization
                
                Return your response as valid JSON with this exact structure:
                {
                    "lambda1": <float>,
                    "max_iter": <int>,
                    "w_threshold": <float>,
                    "reasoning": "<brief explanation of choices>"
                }"""},
                {"role": "user", "content": f"Given these dataset characteristics, what are the optimal NOTEARS hyperparameters?\n\n{context}"}
            ],
            temperature=0.1,
            thinking=False
        )
        
        hyperparams = json.loads(response)
        return hyperparams
        
    except Exception as e:
        print(f"LLM hyperparameter selection failed: {e}")
        # Fallback to rule-based selection
        return _select_hyperparameters_fallback(n_samples, n_features, avg_correlation, sparsity_indicator)

def _select_hyperparameters_fallback(n_samples, n_features, avg_correlation, sparsity_indicator):
    """Fallback rule-based hyperparameter selection."""
    
    # Lambda1 (L1 regularization)
    if n_samples < 100:
        lambda1 = 0.2
    elif n_samples < 500:
        lambda1 = 0.1
    else:
        lambda1 = 0.05
    
    # Adjust for dimensionality
    if n_features > 20:
        lambda1 *= 2
    
    # Adjust for correlation
    if avg_correlation > 0.5:
        lambda1 *= 0.5  # Can use less regularization when strong correlations exist
    elif avg_correlation < 0.2:
        lambda1 *= 2    # Need more regularization for weak correlations
    
    # Max iterations
    if avg_correlation > 0.4 or n_features > 15:
        max_iter = 200
    else:
        max_iter = 100
    
    # Weight threshold
    if n_samples < 200:
        w_threshold = 0.2
    elif sparsity_indicator > 0.5:
        w_threshold = 0.4
    else:
        w_threshold = 0.3
    
    return {
        "lambda1": lambda1,
        "max_iter": max_iter,
        "w_threshold": w_threshold,
        "reasoning": "Fallback rule-based selection"
    }

def run(df, use_gpu=False, fast_mode=None):
    """
    Run NOTEARS with LLM-optimized hyperparameters and optional GPU acceleration.
    """
    
    # Detect GPU capabilities and determine if GPU should be used
    gpu_info = _detect_gpu_capability()
    data_array = df.values if isinstance(df, pd.DataFrame) else df
    should_use_gpu, gpu_reason = _should_use_gpu(data_array.shape, gpu_info)
    
    # Auto-detect fast mode for large datasets
    if fast_mode is None:
        n_samples, n_features = data_array.shape
        fast_mode = (n_features > 20) or (n_samples * n_features > 60000)
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
    
    # Adjust hyperparameters for fast mode
    if fast_mode:
        original_iter = hyperparams["max_iter"]
        hyperparams["max_iter"] = max(50, hyperparams["max_iter"] // 2)  # Reduce iterations
        print(f"Fast mode: Adjusted max_iter from {original_iter} to {hyperparams['max_iter']}")
    
    print(f"NOTEARS Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")

    if use_gpu and TORCH_AVAILABLE:
        # Use GPU-accelerated NOTEARS implementation
        try:
            gpu_notears = GPUAcceleratedNOTEARS()
            gpu_notears.fast_mode = fast_mode
            
            W = gpu_notears.run_notears_algorithm(
                data_array,
                lambda1=hyperparams["lambda1"],
                max_iter=hyperparams["max_iter"]
            )
            
            hyperparams["gpu_accelerated"] = True
            hyperparams["gpu_backend"] = "pytorch"
            hyperparams["fast_mode"] = fast_mode
            
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fallback to CPU version
            W = notears_linear(
                data_array,
                lambda1=hyperparams["lambda1"],
                max_iter=hyperparams["max_iter"],
                loss_type='l2'
            )
            hyperparams["gpu_accelerated"] = False
            hyperparams["fast_mode"] = fast_mode
    else:
        # Run standard CPU version
        print("Running CPU-based NOTEARS algorithm...")
        if fast_mode:
            print("Fast mode enabled: Using reduced iterations")
        
        W = notears_linear(
            data_array,
            lambda1=hyperparams["lambda1"],
            max_iter=hyperparams["max_iter"],
            loss_type='l2'
        )
        hyperparams["gpu_accelerated"] = False
        hyperparams["fast_mode"] = fast_mode

    dag = nx.DiGraph()
    dag.add_nodes_from(df.columns)

    # Apply optimized weight threshold
    w_threshold = hyperparams["w_threshold"]
    
    for i, src in enumerate(df.columns):
        for j, tgt in enumerate(df.columns):
            w = W[i, j]
            if abs(w) > w_threshold:
                dag.add_edge(src, tgt, weight=float(w))

    # Store hyperparameters in graph metadata
    dag.graph['notears_hyperparameters'] = hyperparams
    
    return dag
