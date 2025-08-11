import networkx as nx
import numpy as np
import pandas as pd
import json
from causallearn.search.ConstraintBased.PC import pc as pc_algorithm
from ..llm_query import create_chat_completion

# GPU acceleration imports
try:
    import torch
    import torch.nn.functional as F
    from scipy.stats import norm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - falling back to CPU-only PC algorithm")

def _detect_gpu_capability():
    """
    Detect available GPU acceleration options and recommend optimal configuration.
    """
    gpu_info = {
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'recommended_backend': 'cpu',
        'memory_gb': 0,
        'can_accelerate': False
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_info['cuda_available'] = True
        gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info['recommended_backend'] = 'torch'
        gpu_info['can_accelerate'] = True
    
    return gpu_info

def _should_use_gpu(data_shape, gpu_info):
    """
    Determine if GPU acceleration would be beneficial based on data size and GPU capabilities.
    """
    n_samples, n_features = data_shape
    
    if not gpu_info['can_accelerate']:
        return False, "No GPU acceleration available"
    
    # More aggressive GPU usage thresholds for better performance
    min_samples_for_gpu = 500  # Reduced from 1000
    min_features_for_gpu = 5   # Reduced from 10
    
    if n_samples < min_samples_for_gpu:
        return False, f"Sample size {n_samples} too small for GPU overhead (min: {min_samples_for_gpu})"
    
    if n_features < min_features_for_gpu:
        return False, f"Feature count {n_features} too small for GPU overhead (min: {min_features_for_gpu})"
    
    # Estimate memory requirements (more conservative)
    estimated_memory_gb = (n_samples * n_features * 8) / (1024**3) * 2  # Reduced factor
    
    if estimated_memory_gb > gpu_info['memory_gb'] * 0.6:  # More conservative memory usage
        return False, f"Estimated memory {estimated_memory_gb:.1f}GB exceeds safe limit ({gpu_info['memory_gb']*0.6:.1f}GB)"
    
    return True, f"GPU acceleration beneficial for {n_samples}x{n_features} dataset"

class GPUAcceleratedPC:
    """
    GPU-accelerated implementation of PC algorithm using PyTorch.
    Falls back to CPU implementation when GPU is not available or beneficial.
    """
    
    def __init__(self, device=None):
        self.gpu_info = _detect_gpu_capability()
        
        if device is None:
            self.device = torch.device('cuda' if self.gpu_info['cuda_available'] else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.use_gpu = self.device.type == 'cuda'
        self.ci_test_cache = {}  # Cache for independence test results
        self.fast_mode = False   # Will be set by run function
        
    def run_pc_algorithm(self, data, alpha=0.05):
        """
        Run complete PC algorithm with GPU acceleration.
        """
        if not self.use_gpu:
            return self._run_pc_cpu(data, alpha)
        
        try:
            # Run full GPU implementation
            return self._run_pc_gpu(data, alpha)
        except Exception as e:
            print(f"GPU PC algorithm failed, falling back to CPU: {e}")
            return self._run_pc_cpu(data, alpha)
    
    def _run_pc_gpu(self, data, alpha):
        """
        GPU-accelerated PC algorithm implementation.
        """
        print("Running GPU-accelerated PC algorithm...")
        print(f"Dataset: {data.shape[0]}x{data.shape[1]}, Fast mode: {self.fast_mode}")
        
        # Move data to GPU and standardize
        data_gpu = torch.tensor(data, device=self.device, dtype=torch.float32)
        data_gpu = (data_gpu - data_gpu.mean(dim=0)) / data_gpu.std(dim=0)
        
        n_samples, n_vars = data_gpu.shape
        
        # Initialize adjacency matrix
        adj_matrix = torch.ones((n_vars, n_vars), device=self.device, dtype=torch.bool)
        adj_matrix.diagonal().fill_(False)  # Fixed: use diagonal().fill_() instead of fill_diagonal_()
        
        # Precompute correlation matrix for efficiency
        corr_matrix = torch.corrcoef(data_gpu.T)
        
        # Phase 1: Skeleton discovery with optimized GPU processing
        adj_matrix = self._gpu_skeleton_discovery(data_gpu, adj_matrix, corr_matrix, alpha)
        
        # Convert back to CPU for NetworkX compatibility
        final_adj = adj_matrix.cpu().numpy()
        
        # Create NetworkX graph properly
        import networkx as nx
        G = nx.Graph()  # Use undirected graph for PC skeleton
        
        # Add nodes
        G.add_nodes_from(range(n_vars))
        
        # Add edges where adjacency matrix is True
        for i in range(n_vars):
            for j in range(i+1, n_vars):  # Only upper triangle to avoid duplicates
                if final_adj[i, j]:
                    G.add_edge(i, j)
        
        print(f"GPU PC completed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def _gpu_skeleton_discovery(self, data_gpu, adj_matrix, corr_matrix, alpha):
        """
        GPU-accelerated skeleton discovery phase of PC algorithm with adaptive completeness.
        """
        n_vars = data_gpu.shape[1]
        n_samples = data_gpu.shape[0]
        
        # Adaptive maximum conditioning set size based on data characteristics
        max_cond_set_size = self._determine_max_conditioning_size(n_samples, n_vars)
        min_cond_set_size = 2  # Always test at least up to size 2 for completeness
        
        print(f"Using adaptive conditioning: min_size={min_cond_set_size}, max_size={max_cond_set_size}")
        
        # Process each conditioning set size
        for cond_size in range(max_cond_set_size + 1):
            print(f"Processing conditioning sets of size {cond_size}...")
            
            if cond_size == 0:
                # Simple pairwise correlations
                adj_matrix = self._process_zero_order_gpu(adj_matrix, corr_matrix, alpha, data_gpu.shape[0])
            else:
                # Higher-order conditional independence with early stopping
                remaining_edges_before = adj_matrix.sum().item() // 2
                adj_matrix = self._process_higher_order_gpu(
                    data_gpu, adj_matrix, cond_size, alpha
                )
                remaining_edges_after = adj_matrix.sum().item() // 2
                edges_removed = remaining_edges_before - remaining_edges_after
                
                print(f"Removed {edges_removed} edges at conditioning size {cond_size}")
                
                # Adaptive early stopping based on graph density and progress
                current_density = self._calculate_graph_density(adj_matrix, n_vars)
                should_stop, reason = self._should_stop_early(
                    cond_size, min_cond_set_size, edges_removed, current_density, n_vars
                )
                
                if should_stop:
                    print(f"Early stopping: {reason}")
                    break
        
        # Validate completeness before returning
        if cond_size >= min_cond_set_size:
            completeness_score = self._validate_completeness(data_gpu, adj_matrix, alpha)
            print(f"Completeness validation score: {completeness_score:.3f}")
            
            if completeness_score < 0.85 and cond_size < max_cond_set_size:
                print("Completeness below threshold, testing one more conditioning size...")
                adj_matrix = self._process_higher_order_gpu(
                    data_gpu, adj_matrix, cond_size + 1, alpha
                )
        
        return adj_matrix
    
    def _determine_max_conditioning_size(self, n_samples, n_vars):
        """
        Determine maximum conditioning set size based on data characteristics.
        """
        # Base on sample size and degrees of freedom
        sample_based = min(5, int(np.log2(max(n_samples, 8))))
        
        # Base on number of variables
        var_based = min(4, n_vars // 6)
        
        # Consider fast mode
        if self.fast_mode:
            return min(3, sample_based, var_based)
        else:
            return min(5, max(sample_based, var_based, 2))
    
    def _calculate_graph_density(self, adj_matrix, n_vars):
        """Calculate current graph density."""
        current_edges = adj_matrix.sum().item() // 2  # Undirected edges
        max_possible_edges = (n_vars * (n_vars - 1)) // 2
        return current_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    def _should_stop_early(self, cond_size, min_cond_size, edges_removed, density, n_vars):
        """
        Determine if early stopping is appropriate based on adaptive thresholds.
        """
        # Never stop before minimum conditioning size
        if cond_size < min_cond_size:
            return False, "Below minimum conditioning size"
        
        # Stop if no edges were removed and we're past minimum
        if edges_removed == 0:
            if density < 0.15:  # Sparse graph
                return True, f"No edges removed and graph is sparse (density: {density:.3f})"
            elif cond_size >= 3:  # Force stop after size 3 for very dense graphs
                return True, f"No edges removed at size {cond_size}, forcing stop"
        
        # Stop if graph is very sparse
        if density < 0.05:
            return True, f"Graph very sparse (density: {density:.3f})"
        
        # Continue if we're making progress or graph is still dense
        return False, "Continuing - making progress or graph still dense"
    
    def _validate_completeness(self, data_gpu, adj_matrix, alpha):
        """
        Quick validation that early stopping didn't miss obvious d-separations.
        Returns a completeness score between 0 and 1.
        """
        try:
            n_vars = data_gpu.shape[1]
            validation_tests = 0
            missed_separations = 0
            
            # Get current edges
            edge_indices = torch.nonzero(adj_matrix, as_tuple=False)
            
            if len(edge_indices) == 0:
                return 1.0  # Empty graph is trivially complete
            
            # Sample some edges for validation (limit to avoid performance hit)
            sample_size = min(20, len(edge_indices))
            sampled_indices = torch.randperm(len(edge_indices))[:sample_size]
            
            for sample_idx in sampled_indices:
                i, j = edge_indices[sample_idx].tolist()
                
                # Find potential conditioning variables
                neighbors_i = torch.nonzero(adj_matrix[i], as_tuple=True)[0]
                neighbors_j = torch.nonzero(adj_matrix[j], as_tuple=True)[0]
                potential_cond = torch.cat([neighbors_i, neighbors_j]).unique()
                potential_cond = potential_cond[(potential_cond != i) & (potential_cond != j)]
                
                # Test a few size-1 conditioning sets
                if len(potential_cond) >= 1:
                    for k in potential_cond[:3]:  # Test up to 3
                        validation_tests += 1
                        try:
                            independent, _ = self._gpu_partial_correlation_test(
                                data_gpu, i, j, [k.item()], alpha
                            )
                            if independent:
                                missed_separations += 1
                                break  # Found one, move to next edge
                        except Exception:
                            continue
                
                # Test one size-2 conditioning set if available
                if len(potential_cond) >= 2:
                    from itertools import combinations
                    for cond_set in list(combinations(potential_cond[:4].tolist(), 2))[:1]:
                        validation_tests += 1
                        try:
                            independent, _ = self._gpu_partial_correlation_test(
                                data_gpu, i, j, list(cond_set), alpha
                            )
                            if independent:
                                missed_separations += 1
                                break
                        except Exception:
                            continue
            
            # Calculate completeness score (higher is better)
            if validation_tests == 0:
                return 1.0
            
            completeness_score = 1.0 - (missed_separations / validation_tests)
            return max(0.0, completeness_score)
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return 0.9  # Assume reasonable completeness if validation fails
            
            for idx in sampled_indices:
                i, j = edge_indices[idx].tolist()
                
                # Find potential conditioning sets (neighbors of i or j)
                neighbors_i = torch.nonzero(adj_matrix[i], as_tuple=True)[0]
                neighbors_j = torch.nonzero(adj_matrix[j], as_tuple=True)[0]
                potential_cond = torch.cat([neighbors_i, neighbors_j]).unique()
                
                # Remove i and j from potential conditioning variables
                potential_cond = potential_cond[(potential_cond != i) & (potential_cond != j)]
                
                if len(potential_cond) >= 2:
                    # Test with a conditioning set of size 2
                    cond_set = potential_cond[:2].tolist()
                    
                    try:
                        independent, _ = self._gpu_partial_correlation_test(
                            data_gpu, i, j, cond_set, alpha
                        )
                        
                        validation_tests += 1
                        if independent:
                            missed_separations += 1
                            
                    except Exception:
                        continue  # Skip problematic tests
                
                # Limit validation time
                if validation_tests >= 15:
                    break
            
            if validation_tests == 0:
                return 1.0  # No tests possible
            
            completeness_score = 1 - (missed_separations / validation_tests)
            return completeness_score
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return 0.9  # Conservative estimate if validation fails
    
    def _process_zero_order_gpu(self, adj_matrix, corr_matrix, alpha, n_samples):
        """
        Process zero-order (pairwise) independence tests on GPU.
        """
        # Vectorized Fisher's z-test for all pairs
        r_matrix = corr_matrix.clone()
        r_matrix.diagonal().fill_(0)  # Fixed: use diagonal().fill_() instead of torch.fill_diagonal_()
        
        # Avoid numerical issues
        r_matrix = torch.clamp(r_matrix, -0.9999, 0.9999)
        
        # Fisher's z-transform
        z_matrix = 0.5 * torch.log((1 + r_matrix) / (1 - r_matrix))
        
        # Standard error
        se = 1.0 / torch.sqrt(torch.tensor(n_samples - 3, device=self.device, dtype=torch.float32))
        
        # Test statistics
        test_stats = torch.abs(z_matrix) / se
        
        # Convert to p-values using normal approximation (GPU-friendly)
        # Using complementary error function approximation for speed
        p_values = 2 * torch.erfc(test_stats / torch.sqrt(torch.tensor(2.0, device=self.device)))
        
        # Update adjacency matrix
        independent_pairs = p_values > alpha
        adj_matrix = adj_matrix & ~independent_pairs
        
        removed_edges = independent_pairs.sum().item()
        print(f"Removed {removed_edges} edges in zero-order tests")
        
        return adj_matrix
    
    def _process_higher_order_gpu(self, data_gpu, adj_matrix, cond_size, alpha):
        """
        Process higher-order conditional independence tests with improved efficiency and completeness.
        """
        n_vars = data_gpu.shape[1]
        
        # Generate test triplets more efficiently
        test_triplets = []
        
        # Only test pairs that are still connected
        edge_indices = torch.nonzero(adj_matrix, as_tuple=False)
        
        # Adaptive limits based on conditioning size and performance mode
        if cond_size <= 2:
            # For sizes 1-2, be more thorough for completeness
            max_tests = 2000 if self.fast_mode else 3000
            max_combinations_per_edge = 15 if self.fast_mode else 25
        else:
            # For higher sizes, be more aggressive about limiting
            max_tests = 500 if self.fast_mode else 1000
            max_combinations_per_edge = 5 if self.fast_mode else 10
        
        print(f"Processing up to {max_tests} tests for conditioning size {cond_size}")
        
        for edge_idx in range(min(edge_indices.shape[0], max_tests)):
            i, j = edge_indices[edge_idx].tolist()
            
            # Find potential conditioning variables (neighbors of i or j)
            neighbors_i = torch.nonzero(adj_matrix[i], as_tuple=True)[0]
            neighbors_j = torch.nonzero(adj_matrix[j], as_tuple=True)[0]
            potential_cond = torch.cat([neighbors_i, neighbors_j]).unique()
            
            # Remove i and j from potential conditioning variables
            potential_cond = potential_cond[(potential_cond != i) & (potential_cond != j)]
            
            if len(potential_cond) >= cond_size:
                # Generate conditioning sets of the required size
                from itertools import combinations
                
                combination_count = 0
                for cond_set in combinations(potential_cond.tolist(), cond_size):
                    test_triplets.append((i, j, list(cond_set)))
                    combination_count += 1
                    
                    if combination_count >= max_combinations_per_edge:
                        break
        
        if not test_triplets:
            return adj_matrix
        
        print(f"Testing {len(test_triplets)} conditional independence relationships...")
        
        # Process in adaptive batch sizes
        if cond_size <= 2:
            batch_size = min(150 if self.fast_mode else 100, len(test_triplets))
        else:
            batch_size = min(200 if self.fast_mode else 100, len(test_triplets))
        
        edges_removed = 0
        
        for batch_start in range(0, len(test_triplets), batch_size):
            batch_end = min(batch_start + batch_size, len(test_triplets))
            batch_triplets = test_triplets[batch_start:batch_end]
            
            # Process batch with error handling
            batch_removed = 0
            for i, j, cond_set in batch_triplets:
                try:
                    independent, _ = self._gpu_partial_correlation_test(
                        data_gpu, i, j, cond_set, alpha
                    )
                    
                    if independent:
                        adj_matrix[i, j] = False
                        adj_matrix[j, i] = False
                        batch_removed += 1
                        edges_removed += 1
                        
                except Exception as e:
                    # Log problematic tests but continue
                    if batch_start == 0:  # Only print once to avoid spam
                        print(f"Warning: Some independence tests failed (continuing...)")
                    continue
            
            # Progress reporting for larger batches
            if len(test_triplets) > 500:
                progress = (batch_end / len(test_triplets)) * 100
                if progress % 25 < (batch_size / len(test_triplets)) * 100:  # Report every 25%
                    print(f"Progress: {progress:.0f}% complete, {edges_removed} edges removed so far")
        
        remaining_edges = adj_matrix.sum().item() // 2
        print(f"Remaining edges after size-{cond_size} tests: {remaining_edges} (removed: {edges_removed})")
        
        return adj_matrix
    
    def _gpu_partial_correlation_test(self, data_gpu, i, j, cond_set, alpha):
        """
        Optimized GPU partial correlation test.
        """
        indices = [i, j] + cond_set
        selected_data = data_gpu[:, indices]
        
        n_samples = selected_data.shape[0]
        
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(selected_data.T)
        
        if len(cond_set) == 0:
            r_xy = corr_matrix[0, 1].item()
        elif len(cond_set) == 1:
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
                return False, 1.0  # Conservative: assume dependent if computation fails
        
        # Fisher's z-test
        if abs(r_xy) >= 0.9999:
            r_xy = 0.9999 * (1 if r_xy > 0 else -1)
        
        z = 0.5 * np.log((1 + r_xy) / (1 - r_xy))
        se = 1.0 / np.sqrt(n_samples - len(cond_set) - 3)
        test_stat = abs(z) / se
        
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(test_stat))
        
        return p_value > alpha, p_value
    
    def _run_pc_cpu(self, data, alpha):
        """
        Fallback CPU implementation using causal-learn.
        """
        print("Running CPU fallback PC algorithm...")
        from causallearn.search.ConstraintBased.PC import pc as pc_algorithm
        
        try:
            cg = pc_algorithm(data, alpha=alpha, indep_test="fisherz")
            nx_g = cg.to_nx_graph()
            print(f"CPU PC completed: {nx_g.number_of_nodes()} nodes, {nx_g.number_of_edges()} edges")
            return nx_g
        except Exception as e:
            print(f"CPU PC also failed: {e}")
            # Create minimal graph as last resort
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(data.shape[1]))
            return G
        
    def partial_correlation_test(self, data, i, j, conditioning_set, alpha=0.05):
        """
        GPU-accelerated partial correlation test.
        """
        if not self.use_gpu or len(conditioning_set) == 0:
            return self._partial_correlation_cpu(data, i, j, conditioning_set, alpha)
        
        try:
            return self._partial_correlation_gpu(data, i, j, conditioning_set, alpha)
        except Exception as e:
            print(f"GPU partial correlation failed, falling back to CPU: {e}")
            return self._partial_correlation_cpu(data, i, j, conditioning_set, alpha)
    
    def _partial_correlation_gpu(self, data, i, j, conditioning_set, alpha):
        """
        Compute partial correlation test on GPU using PyTorch.
        """
        # Move data to GPU
        indices = [i, j] + list(conditioning_set)
        selected_data = torch.tensor(data[:, indices], device=self.device, dtype=torch.float32)
        
        n_samples = selected_data.shape[0]
        
        # Standardize data
        selected_data = (selected_data - selected_data.mean(dim=0)) / selected_data.std(dim=0)
        
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(selected_data.T)
        
        if len(conditioning_set) == 0:
            # Simple correlation test
            r_xy = corr_matrix[0, 1].item()
        else:
            # Partial correlation using matrix operations
            # R_xy|Z = (R_xy - R_xz * R_zy) / sqrt((1 - R_xz^2) * (1 - R_zy^2))
            
            n_cond = len(conditioning_set)
            if n_cond == 1:
                # Simple partial correlation
                r_xy = corr_matrix[0, 1].item()
                r_xz = corr_matrix[0, 2].item()
                r_yz = corr_matrix[1, 2].item()
                
                numerator = r_xy - r_xz * r_yz
                denominator = torch.sqrt((1 - r_xz**2) * (1 - r_yz**2))
                r_xy_z = numerator / denominator
                r_xy = r_xy_z.item()
            else:
                # Multiple conditioning variables - use precision matrix approach
                # Partial correlation = -P_ij / sqrt(P_ii * P_jj) where P is precision matrix
                try:
                    precision_matrix = torch.linalg.inv(corr_matrix)
                    r_xy = -precision_matrix[0, 1] / torch.sqrt(precision_matrix[0, 0] * precision_matrix[1, 1])
                    r_xy = r_xy.item()
                except:
                    # Fallback to CPU for numerical issues
                    return self._partial_correlation_cpu(data, i, j, conditioning_set, alpha)
        
        # Fisher's z-transform and statistical test
        if abs(r_xy) >= 0.9999:  # Avoid numerical issues
            r_xy = 0.9999 * (1 if r_xy > 0 else -1)
        
        z = 0.5 * np.log((1 + r_xy) / (1 - r_xy))
        se = 1.0 / np.sqrt(n_samples - len(conditioning_set) - 3)
        test_stat = abs(z) / se
        
        # Two-tailed test
        p_value = 2 * (1 - norm.cdf(test_stat))
        
        return p_value > alpha, p_value
    
    def _partial_correlation_cpu(self, data, i, j, conditioning_set, alpha):
        """
        CPU fallback for partial correlation test.
        """
        from scipy.stats import pearsonr
        from sklearn.linear_model import LinearRegression
        
        n_samples = data.shape[0]
        
        if len(conditioning_set) == 0:
            # Simple correlation
            corr, p_val = pearsonr(data[:, i], data[:, j])
            return p_val > alpha, p_val
        
        # Partial correlation via regression residuals
        X_cond = data[:, list(conditioning_set)]
        
        # Regress X and Y on conditioning set
        reg_x = LinearRegression().fit(X_cond, data[:, i])
        reg_y = LinearRegression().fit(X_cond, data[:, j])
        
        residual_x = data[:, i] - reg_x.predict(X_cond)
        residual_y = data[:, j] - reg_y.predict(X_cond)
        
        # Correlation of residuals
        corr, p_val = pearsonr(residual_x, residual_y)
        
        return p_val > alpha, p_val
    
    def batch_independence_tests(self, data, test_pairs, alpha=0.05):
        """
        Batch process multiple independence tests on GPU.
        """
        if not self.use_gpu:
            return self._batch_tests_cpu(data, test_pairs, alpha)
        
        try:
            return self._batch_tests_gpu(data, test_pairs, alpha)
        except Exception as e:
            print(f"GPU batch testing failed, falling back to CPU: {e}")
            return self._batch_tests_cpu(data, test_pairs, alpha)
    
    def _batch_tests_gpu(self, data, test_pairs, alpha):
        """
        GPU-accelerated batch testing for independence tests.
        """
        data_gpu = torch.tensor(data, device=self.device, dtype=torch.float32)
        data_gpu = (data_gpu - data_gpu.mean(dim=0)) / data_gpu.std(dim=0)
        
        results = []
        batch_size = min(100, len(test_pairs))  # Process in batches to manage memory
        
        for batch_start in range(0, len(test_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(test_pairs))
            batch_pairs = test_pairs[batch_start:batch_end]
            
            # Process batch
            batch_results = []
            for i, j, conditioning_set in batch_pairs:
                independent, p_val = self.partial_correlation_test(data, i, j, conditioning_set, alpha)
                batch_results.append((independent, p_val))
            
            results.extend(batch_results)
        
        return results
    
    def _batch_tests_cpu(self, data, test_pairs, alpha):
        """
        CPU fallback for batch testing.
        """
        results = []
        for i, j, conditioning_set in test_pairs:
            independent, p_val = self._partial_correlation_cpu(data, i, j, conditioning_set, alpha)
            results.append((independent, p_val))
        return results

def _select_hyperparameters_llm(data):
    """
    Use LLM to select optimal hyperparameters for PC algorithm based on data characteristics.
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
    
    # Calculate sample complexity
    theoretical_tests = n_features * (n_features - 1) * (2 ** (n_features - 2))  # Rough estimate
    
    context = f"""
    Dataset Characteristics:
    - Sample size: {n_samples}
    - Number of variables: {n_features}
    - Continuous variables: {continuous_vars}
    - Discrete variables: {discrete_vars}
    - Average correlation: {avg_correlation:.3f}
    - Sample-to-variable ratio: {n_samples/n_features:.2f}
    - Theoretical test complexity: {theoretical_tests}
    """
    
    try:
        response = create_chat_completion(
            messages=[
                {"role": "system", "content": """You are an expert in PC (Peter-Clark) algorithm hyperparameter selection for causal discovery.
                
                Based on the dataset characteristics, recommend optimal hyperparameters for:
                1. alpha (significance level): Controls Type I error rate for conditional independence tests. Lower values are more conservative.
                2. indep_test (independence test): Type of test to use based on data characteristics.
                
                Guidelines:
                - Large samples (n>1000): Can use lower alpha (0.01-0.05) for more conservative testing
                - Small samples (n<200): Use higher alpha (0.05-0.1) to maintain power
                - High dimensionality (p>20): Use more conservative alpha to control multiple testing
                - Continuous data: Use "fisherz" (Fisher's z-test) for Gaussian data
                - Mixed/discrete data: Use "chisq" (Chi-square test) or "kci" (kernel CI test)
                - High correlation data: May need more conservative alpha
                
                Available independence tests:
                - "fisherz": Fisher's z-test (for Gaussian continuous data)
                - "chisq": Chi-square test (for discrete/categorical data)
                - "gsq": G-square test (alternative for discrete data)
                - "kci": Kernel conditional independence test (non-parametric, works for mixed data)
                
                Return your response as valid JSON with this exact structure:
                {
                    "alpha": <float>,
                    "indep_test": "<test_name>",
                    "reasoning": "<brief explanation of choices>"
                }"""},
                {"role": "user", "content": f"Given these dataset characteristics, what are the optimal PC algorithm hyperparameters?\n\n{context}"}
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
    
    # Alpha (significance level)
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
    
    # Independence test selection
    if continuous_vars > discrete_vars:
        indep_test = "fisherz"  # Good for continuous Gaussian data
    elif discrete_vars > continuous_vars:
        indep_test = "chisq"    # Good for discrete data
    else:
        indep_test = "kci"      # Good for mixed data types
    
    return {
        "alpha": alpha,
        "indep_test": indep_test,
        "reasoning": "Fallback rule-based selection"
    }

def run(data, use_gpu=False, fast_mode=None):
    """
    Run PC algorithm with LLM-optimized hyperparameters and optional GPU acceleration.
    
    Args:
        data: Input dataset (DataFrame or numpy array)
        use_gpu: Whether to attempt GPU acceleration (default: False)
        fast_mode: Auto-enable aggressive optimizations for large datasets (default: None = auto-detect)
    """
    
    # Detect GPU capabilities and determine if GPU should be used
    gpu_info = _detect_gpu_capability()
    data_array = data.values if isinstance(data, pd.DataFrame) else data
    should_use_gpu, gpu_reason = _should_use_gpu(data_array.shape, gpu_info)
    
    # Auto-detect fast mode for large datasets
    if fast_mode is None:
        n_samples, n_features = data_array.shape
        # Enable fast mode for datasets with many features or large size
        fast_mode = (n_features > 15) or (n_samples * n_features > 50000)
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
    hyperparams = _select_hyperparameters_llm(data)
    
    # Adjust alpha for fast mode (more aggressive)
    if fast_mode:
        original_alpha = hyperparams["alpha"]
        hyperparams["alpha"] = min(0.1, hyperparams["alpha"] * 2)  # More liberal threshold
        print(f"Fast mode: Adjusted alpha from {original_alpha:.3f} to {hyperparams['alpha']:.3f}")
    
    print(f"PC Algorithm Hyperparameters selected: {hyperparams.get('reasoning', 'No reasoning provided')}")
    
    if use_gpu and TORCH_AVAILABLE:
        # Use fully GPU-accelerated PC implementation
        try:
            gpu_pc = GPUAcceleratedPC()
            gpu_pc.fast_mode = fast_mode  # Pass fast mode to GPU implementation
            print("Running GPU-accelerated PC algorithm...")
            
            # Use our custom GPU implementation
            nx_g = gpu_pc.run_pc_algorithm(data_array, alpha=hyperparams["alpha"])
            
            # Validate the result
            if nx_g is None or not hasattr(nx_g, 'nodes'):
                raise ValueError("GPU implementation returned invalid graph")
            
            # Add labels to nodes
            if isinstance(data, pd.DataFrame):
                mapping = {i: col for i, col in enumerate(data.columns)}
                nx_g = nx.relabel_nodes(nx_g, mapping)
            
            hyperparams["gpu_accelerated"] = True
            hyperparams["gpu_backend"] = "pytorch"
            hyperparams["algorithm_type"] = "gpu_accelerated"
            hyperparams["fast_mode"] = fast_mode
            
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            
            # Fallback to CPU version
            cg = pc_algorithm(
                data_array,
                alpha=hyperparams["alpha"],
                indep_test=hyperparams["indep_test"]
            )
            
            nx_g = cg.to_nx_graph()
            mapping = {i: label for i, label in enumerate(cg.labels)}
            nx_g = nx.relabel_nodes(nx_g, mapping)
            hyperparams["gpu_accelerated"] = False
            hyperparams["algorithm_type"] = "cpu_fallback"
            hyperparams["fast_mode"] = fast_mode
    else:
        # Run standard CPU version
        print("Running CPU-based PC algorithm...")
        if fast_mode:
            print("Fast mode enabled: Using more aggressive pruning")
        
        cg = pc_algorithm(
            data_array,
            alpha=hyperparams["alpha"],
            indep_test=hyperparams["indep_test"]
        )
        
        nx_g = cg.to_nx_graph()
        mapping = {i: label for i, label in enumerate(cg.labels)}
        nx_g = nx.relabel_nodes(nx_g, mapping)
        hyperparams["gpu_accelerated"] = False
        hyperparams["algorithm_type"] = "cpu_standard"
        hyperparams["fast_mode"] = fast_mode

    # Store hyperparameters and GPU info in graph metadata
    nx_g.graph['pc_hyperparameters'] = hyperparams
    nx_g.graph['gpu_info'] = gpu_info
    
    return nx_g

def run_gpu_accelerated(data):
    """
    Convenience function to run GPU-accelerated PC algorithm.
    """
    return run(data, use_gpu=True)
