"""
Population Structure Analysis for PyGWAS
Implements PCA, kinship matrix calculation, and population stratification methods
"""

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union, Dict
import warnings
from .utils import standardize_genotypes

class PopulationStructure:
    """
    Population structure analysis methods
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize PopulationStructure object
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
    def calculate_pca(self, genotypes: np.ndarray, 
                     n_components: int = 10,
                     standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Principal Components Analysis on genotype data
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            n_components: Number of principal components to calculate
            standardize: Whether to standardize genotypes before PCA
            
        Returns:
            PC scores (samples x components) and explained variance ratios
        """
        if self.verbose:
            print(f"Calculating PCA with {n_components} components...")
            
        # Handle missing data and standardize
        if standardize:
            geno_std = standardize_genotypes(genotypes, method='unit_variance')
        else:
            geno_std = genotypes.copy()
            # Replace missing with mean
            for i in range(genotypes.shape[1]):
                col = genotypes[:, i]
                valid_mask = ~np.isnan(col)
                if np.sum(valid_mask) > 0:
                    mean_val = np.mean(col[valid_mask])
                    geno_std[~valid_mask, i] = mean_val
        
        # Remove SNPs with zero variance
        var_mask = np.var(geno_std, axis=0) > 0
        geno_std = geno_std[:, var_mask]
        
        if geno_std.shape[1] == 0:
            raise ValueError("No SNPs remaining after variance filtering")
            
        # Perform PCA
        n_components = min(n_components, min(geno_std.shape) - 1)
        
        try:
            pca = PCA(n_components=n_components, random_state=42)
            pc_scores = pca.fit_transform(geno_std)
            explained_variance = pca.explained_variance_ratio_
            
        except Exception as e:
            # Fallback to SVD if PCA fails
            warnings.warn(f"PCA failed, using SVD: {e}")
            U, s, Vt = np.linalg.svd(geno_std - np.mean(geno_std, axis=0), full_matrices=False)
            pc_scores = U[:, :n_components] * s[:n_components]
            total_var = np.sum(s**2)
            explained_variance = (s[:n_components]**2) / total_var
            
        if self.verbose:
            print(f"PCA complete. First 3 PCs explain {np.sum(explained_variance[:3])*100:.2f}% of variance")
            
        return pc_scores, explained_variance
    
    def calculate_kinship_matrix(self, genotypes: np.ndarray,
                               method: str = 'vanraden') -> np.ndarray:
        """
        Calculate kinship/relationship matrix
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            method: 'vanraden' or 'gcta'
            
        Returns:
            Kinship matrix (samples x samples)
        """
        if self.verbose:
            print(f"Calculating kinship matrix using {method} method...")
            
        n_samples, n_snps = genotypes.shape
        
        # Standardize genotypes
        geno_std = standardize_genotypes(genotypes, method='unit_variance')
        
        # Remove SNPs with zero variance
        var_mask = np.var(geno_std, axis=0) > 0
        geno_std = geno_std[:, var_mask]
        n_snps_used = geno_std.shape[1]
        
        if method == 'vanraden':
            # VanRaden method: K = XX'/m where X is standardized genotypes
            kinship = np.dot(geno_std, geno_std.T) / n_snps_used
            
        elif method == 'gcta':
            # GCTA method: similar to VanRaden but with different normalization
            kinship = np.dot(geno_std, geno_std.T) / n_snps_used
            
        else:
            raise ValueError(f"Unknown kinship method: {method}")
            
        if self.verbose:
            print(f"Kinship matrix calculated using {n_snps_used} SNPs")
            print(f"Mean kinship: {np.mean(kinship):.6f}, Range: [{np.min(kinship):.6f}, {np.max(kinship):.6f}]")
            
        return kinship
    
    def calculate_grm(self, genotypes: np.ndarray) -> np.ndarray:
        """
        Calculate Genomic Relationship Matrix (GRM)
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            
        Returns:
            GRM matrix (samples x samples)
        """
        return self.calculate_kinship_matrix(genotypes, method='vanraden')
    
    def estimate_effective_population_size(self, genotypes: np.ndarray,
                                         genetic_map: Optional[pd.DataFrame] = None) -> float:
        """
        Estimate effective population size using LD-based methods
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            genetic_map: Genetic map with cM positions (optional)
            
        Returns:
            Estimated effective population size
        """
        if self.verbose:
            print("Estimating effective population size...")
            
        # This is a simplified implementation
        # In practice, would use more sophisticated LD-based methods
        
        # Calculate LD between adjacent SNPs
        n_snps = genotypes.shape[1]
        ld_values = []
        
        for i in range(min(1000, n_snps - 1)):  # Sample first 1000 SNPs for speed
            snp1 = genotypes[:, i]
            snp2 = genotypes[:, i + 1]
            
            # Calculate correlation (r)
            valid_mask = ~(np.isnan(snp1) | np.isnan(snp2))
            if np.sum(valid_mask) > 10:
                r = np.corrcoef(snp1[valid_mask], snp2[valid_mask])[0, 1]
                if not np.isnan(r):
                    ld_values.append(r**2)
        
        if len(ld_values) == 0:
            return np.nan
            
        # Simple Ne estimation: Ne ≈ 1/(4*c*E[r²]) where c is recombination rate
        mean_ld = np.mean(ld_values)
        c = 0.01  # Assume 1 cM between adjacent SNPs
        ne_estimate = 1 / (4 * c * mean_ld)
        
        if self.verbose:
            print(f"Estimated effective population size: {ne_estimate:.0f}")
            
        return ne_estimate
    
    def detect_population_stratification(self, genotypes: np.ndarray,
                                       n_components: int = 10) -> Tuple[np.ndarray, float]:
        """
        Detect population stratification using PCA
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            n_components: Number of PCs to calculate
            
        Returns:
            PC scores and Tracy-Widom test statistic for PC1
        """
        if self.verbose:
            print("Detecting population stratification...")
            
        # Calculate PCA
        pc_scores, explained_var = self.calculate_pca(genotypes, n_components)
        
        # Simple test for population stratification
        # In practice, would use Tracy-Widom test or other methods
        pc1_var = explained_var[0]
        
        # Rule of thumb: PC1 explains >1% of variance suggests stratification
        stratification_threshold = 0.01
        is_stratified = pc1_var > stratification_threshold
        
        if self.verbose:
            print(f"PC1 explains {pc1_var*100:.2f}% of variance")
            if is_stratified:
                print("Population stratification detected")
            else:
                print("No significant population stratification detected")
                
        return pc_scores, pc1_var
    
    def adjust_for_population_structure(self, genotypes: np.ndarray,
                                      phenotypes: np.ndarray,
                                      n_pcs: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate principal components for use as covariates
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            phenotypes: Phenotype values
            n_pcs: Number of PCs to include
            
        Returns:
            PC matrix for use as covariates and PC scores
        """
        if self.verbose:
            print(f"Generating {n_pcs} PCs for population structure adjustment...")
            
        # Calculate PCs
        pc_scores, _ = self.calculate_pca(genotypes, n_components=n_pcs)
        
        # Create covariate matrix
        pc_covariates = pc_scores[:, :n_pcs]
        
        if self.verbose:
            print(f"Generated {pc_covariates.shape[1]} PC covariates")
            
        return pc_covariates, pc_scores
    
    def calculate_fst(self, genotypes1: np.ndarray, genotypes2: np.ndarray) -> np.ndarray:
        """
        Calculate FST between two populations
        
        Args:
            genotypes1: Genotype matrix for population 1
            genotypes2: Genotype matrix for population 2
            
        Returns:
            FST values for each SNP
        """
        if self.verbose:
            print("Calculating FST between populations...")
            
        n_snps = genotypes1.shape[1]
        fst_values = np.zeros(n_snps)
        
        for i in range(n_snps):
            g1 = genotypes1[:, i]
            g2 = genotypes2[:, i]
            
            # Remove missing values
            g1 = g1[~np.isnan(g1)]
            g2 = g2[~np.isnan(g2)]
            
            if len(g1) == 0 or len(g2) == 0:
                fst_values[i] = np.nan
                continue
                
            # Calculate allele frequencies
            p1 = np.mean(g1) / 2
            p2 = np.mean(g2) / 2
            
            # Calculate FST
            n1, n2 = len(g1), len(g2)
            p_total = (n1 * p1 + n2 * p2) / (n1 + n2)
            
            ht = 2 * p_total * (1 - p_total)  # Total heterozygosity
            hs = (n1 * 2 * p1 * (1 - p1) + n2 * 2 * p2 * (1 - p2)) / (n1 + n2)  # Within-pop heterozygosity
            
            if ht > 0:
                fst_values[i] = (ht - hs) / ht
            else:
                fst_values[i] = 0
                
        if self.verbose:
            valid_fst = fst_values[~np.isnan(fst_values)]
            if len(valid_fst) > 0:
                print(f"Mean FST: {np.mean(valid_fst):.4f}, Range: [{np.min(valid_fst):.4f}, {np.max(valid_fst):.4f}]")
            
        return fst_values
    
    def calculate_admixture_proportions(self, genotypes: np.ndarray,
                                      reference_pops: Dict[str, np.ndarray],
                                      method: str = 'least_squares') -> pd.DataFrame:
        """
        Estimate admixture proportions using reference populations
        
        Args:
            genotypes: Target genotype matrix
            reference_pops: Dictionary of reference population genotype matrices
            method: Method for estimation ('least_squares' or 'nnls')
            
        Returns:
            DataFrame with admixture proportions for each sample
        """
        if self.verbose:
            print(f"Estimating admixture proportions using {method} method...")
            
        # This is a simplified implementation
        # In practice, would use more sophisticated methods like ADMIXTURE
        
        n_samples = genotypes.shape[0]
        pop_names = list(reference_pops.keys())
        n_pops = len(pop_names)
        
        # Calculate allele frequencies for each reference population
        ref_freqs = {}
        for pop_name, ref_geno in reference_pops.items():
            freqs = np.nanmean(ref_geno, axis=0) / 2
            ref_freqs[pop_name] = freqs
            
        # Estimate admixture proportions for each sample
        admix_props = np.zeros((n_samples, n_pops))
        
        for i in range(n_samples):
            sample_geno = genotypes[i, :] / 2  # Convert to allele frequencies
            
            # Set up least squares problem
            A = np.array([ref_freqs[pop] for pop in pop_names]).T
            b = sample_geno
            
            # Remove missing values
            valid_mask = ~(np.isnan(b) | np.any(np.isnan(A), axis=1))
            if np.sum(valid_mask) < 10:
                admix_props[i, :] = np.nan
                continue
                
            A_valid = A[valid_mask, :]
            b_valid = b[valid_mask]
            
            try:
                if method == 'least_squares':
                    # Constrained least squares (proportions sum to 1, non-negative)
                    from scipy.optimize import minimize
                    
                    def objective(x):
                        return np.sum((A_valid @ x - b_valid)**2)
                    
                    def constraint(x):
                        return np.sum(x) - 1
                    
                    result = minimize(objective, 
                                    x0=np.ones(n_pops) / n_pops,
                                    bounds=[(0, 1) for _ in range(n_pops)],
                                    constraints={'type': 'eq', 'fun': constraint})
                    
                    if result.success:
                        admix_props[i, :] = result.x
                    else:
                        admix_props[i, :] = np.nan
                        
                elif method == 'nnls':
                    # Non-negative least squares
                    from scipy.optimize import nnls
                    props, _ = nnls(A_valid, b_valid)
                    props = props / np.sum(props) if np.sum(props) > 0 else props
                    admix_props[i, :] = props
                    
            except:
                admix_props[i, :] = np.nan
                
        # Create DataFrame
        admix_df = pd.DataFrame(admix_props, columns=pop_names)
        
        if self.verbose:
            print(f"Admixture estimation complete for {n_samples} samples")
            
        return admix_df
