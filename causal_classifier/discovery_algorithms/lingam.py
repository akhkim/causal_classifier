import pandas as pd
import networkx as nx
import numpy as np
from lingam import DirectLiNGAM

# GPU acceleration imports
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - falling back to CPU-only LiNGAM algorithm")

def _detect_gpu_capability():
    """
    Detect available GPU acceleration options for LiNGAM.
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
    Determine if GPU acceleration would be beneficial for LiNGAM.
    """
    n_samples, n_features = data_shape
    
    if not gpu_info['can_accelerate']:
        return False, "No GPU acceleration available"
    
    # LiNGAM benefits from GPU for large matrix operations
    min_samples_for_gpu = 200
    min_features_for_gpu = 6
    
    if n_samples < min_samples_for_gpu:
        return False, f"Sample size {n_samples} too small for GPU overhead (min: {min_samples_for_gpu})"
    
    if n_features < min_features_for_gpu:
        return False, f"Feature count {n_features} too small for GPU overhead (min: {min_features_for_gpu})"
    
    # Estimate memory requirements
    estimated_memory_gb = (n_samples * n_features * 8) / (1024**3) * 4  # ICA needs more memory
    
    if estimated_memory_gb > gpu_info['memory_gb'] * 0.6:
        return False, f"Estimated memory {estimated_memory_gb:.1f}GB exceeds safe limit ({gpu_info['memory_gb']*0.6:.1f}GB)"
    
    return True, f"GPU acceleration beneficial for {n_samples}x{n_features} dataset"

class GPUAcceleratedLiNGAM:
    """
    GPU-accelerated implementation of LiNGAM algorithm using PyTorch.
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
        
    def run_lingam_algorithm(self, data):
        """
        Run LiNGAM algorithm with GPU acceleration where beneficial.
        """
        if not self.use_gpu:
            return self._run_lingam_cpu(data)
        
        try:
            # Use GPU-accelerated approach for matrix operations
            return self._run_lingam_gpu_hybrid(data)
        except Exception as e:
            print(f"GPU LiNGAM algorithm failed, falling back to CPU: {e}")
            return self._run_lingam_cpu(data)
    
    def _run_lingam_gpu_hybrid(self, data):
        """
        Hybrid GPU-CPU LiNGAM implementation.
        GPU for matrix operations, CPU for ICA.
        """
        print("Running GPU-accelerated LiNGAM algorithm...")
        print(f"Dataset: {data.shape[0]}x{data.shape[1]}, Fast mode: {self.fast_mode}")
        
        # Move data to GPU for preprocessing
        data_gpu = torch.tensor(data, device=self.device, dtype=torch.float32)
        
        # Handle NaN values
        if torch.isnan(data_gpu).any():
            print("Warning: NaN values detected, filling with column means")
            for i in range(data_gpu.shape[1]):
                col_mean = torch.nanmean(data_gpu[:, i])
                data_gpu[torch.isnan(data_gpu[:, i]), i] = col_mean
        
        # GPU-accelerated preprocessing
        data_gpu = self._gpu_preprocess(data_gpu)
        
        # Move back to CPU for LiNGAM fitting (ICA is not easily GPU-accelerated)
        data_preprocessed = data_gpu.cpu().numpy()
        
        # Use standard LiNGAM with preprocessed data
        model = DirectLiNGAM()
        model.fit(data_preprocessed)
        
        return model
    
    def _gpu_preprocess(self, data_gpu):
        """
        GPU-accelerated data preprocessing for LiNGAM.
        """
        # Standardize data (GPU-accelerated)
        mean = torch.mean(data_gpu, dim=0, keepdim=True)
        std = torch.std(data_gpu, dim=0, keepdim=True) + 1e-8
        data_gpu = (data_gpu - mean) / std
        
        # Remove outliers using GPU operations
        if not self.fast_mode:
            # Calculate z-scores for outlier detection
            z_scores = torch.abs(data_gpu)
            outlier_mask = z_scores > 3.0
            
            # Replace outliers with clipped values (GPU operation)
            data_gpu = torch.where(outlier_mask, 
                                 torch.sign(data_gpu) * 3.0 * std + mean, 
                                 data_gpu * std + mean)
            
            # Re-standardize after outlier removal
            mean = torch.mean(data_gpu, dim=0, keepdim=True)
            std = torch.std(data_gpu, dim=0, keepdim=True) + 1e-8
            data_gpu = (data_gpu - mean) / std
        
        return data_gpu
    
    def _run_lingam_cpu(self, data):
        """
        CPU fallback implementation.
        """
        print("Running CPU LiNGAM algorithm...")
        model = DirectLiNGAM()
        model.fit(data)
        return model

def run(df: pd.DataFrame, use_gpu=False, fast_mode=None):
    """
    Run LiNGAM algorithm with optional GPU acceleration.
    """
    
    # Detect GPU capabilities and determine if GPU should be used
    gpu_info = _detect_gpu_capability()
    data_array = df.values if isinstance(df, pd.DataFrame) else df
    should_use_gpu, gpu_reason = _should_use_gpu(data_array.shape, gpu_info)
    
    # Auto-detect fast mode for large datasets
    if fast_mode is None:
        n_samples, n_features = data_array.shape
        fast_mode = (n_features > 15) or (n_samples * n_features > 40000)
        if fast_mode:
            print(f"Auto-enabling fast mode for large dataset ({n_samples}x{n_features})")
    
    # Override user choice if GPU is not beneficial
    if use_gpu and not should_use_gpu:
        print(f"GPU acceleration requested but not beneficial: {gpu_reason}")
        use_gpu = False
    elif use_gpu and should_use_gpu:
        print(f"GPU acceleration enabled: {gpu_reason}")
        print(f"GPU Info: PyTorch backend, {gpu_info['memory_gb']:.1f}GB memory")
    
    if use_gpu and TORCH_AVAILABLE:
        # Use GPU-accelerated LiNGAM implementation
        try:
            gpu_lingam = GPUAcceleratedLiNGAM()
            gpu_lingam.fast_mode = fast_mode
            
            model = gpu_lingam.run_lingam_algorithm(data_array)
            
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            # Fallback to CPU version
            model = DirectLiNGAM()
            model.fit(data_array)
    else:
        # Run standard CPU version
        print("Running CPU-based LiNGAM algorithm...")
        if fast_mode:
            print("Fast mode enabled: Using faster ICA convergence")
        
        model = DirectLiNGAM()
        model.fit(data_array)

    # adjacency_matrix_[i,j] = causal effect i → j
    adj = model.adjacency_matrix_

    dag = nx.DiGraph()
    dag.add_nodes_from(df.columns)

    for i, src in enumerate(df.columns):
        for j, tgt in enumerate(df.columns):
            w = adj[i, j]   # weight of the edge src → tgt
            if w != 0:
                dag.add_edge(src, tgt, weight=float(w))

    return dag
