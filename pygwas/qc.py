"""
Quality Control module for PyGWAS
Implements comprehensive QC procedures matching or exceeding PLINK2 standards
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
import logging
from .utils import (
    calculate_maf, calculate_hwe_pvalue, calculate_call_rate,
    validate_input_data
)

class QualityControl:
    """
    Quality Control class for genotype and phenotype data
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize QC object
        
        Args:
            verbose: Whether to print QC statistics
        """
        self.verbose = verbose
        self.qc_stats = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
    def filter_samples_by_call_rate(self, genotypes: np.ndarray, 
                                   sample_ids: np.ndarray,
                                   min_call_rate: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter samples by genotype call rate
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            sample_ids: Sample identifiers
            min_call_rate: Minimum call rate threshold
            
        Returns:
            Filtered genotypes, sample_ids, and mask of kept samples
        """
        sample_call_rates = calculate_call_rate(genotypes, axis=1)
        keep_samples = sample_call_rates >= min_call_rate
        
        n_removed = np.sum(~keep_samples)
        if self.verbose:
            self.logger.info(f"Sample call rate filter: removed {n_removed} samples "
                           f"({n_removed/len(sample_ids)*100:.2f}%) with call rate < {min_call_rate}")
            
        self.qc_stats['samples_removed_call_rate'] = n_removed
        self.qc_stats['samples_call_rate_threshold'] = min_call_rate
        
        # Ensure sample_ids is a numpy array for boolean indexing
        sample_ids_array = np.array(sample_ids)
        return genotypes[keep_samples, :], sample_ids_array[keep_samples], keep_samples
    
    def filter_snps_by_call_rate(self, genotypes: np.ndarray,
                                 snp_ids: np.ndarray,
                                 min_call_rate: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter SNPs by call rate
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            snp_ids: SNP identifiers
            min_call_rate: Minimum call rate threshold
            
        Returns:
            Filtered genotypes, snp_ids, and mask of kept SNPs
        """
        snp_call_rates = calculate_call_rate(genotypes, axis=0)
        keep_snps = snp_call_rates >= min_call_rate
        
        n_removed = np.sum(~keep_snps)
        if self.verbose:
            self.logger.info(f"SNP call rate filter: removed {n_removed} SNPs "
                           f"({n_removed/len(snp_ids)*100:.2f}%) with call rate < {min_call_rate}")
            
        self.qc_stats['snps_removed_call_rate'] = n_removed
        self.qc_stats['snps_call_rate_threshold'] = min_call_rate
        
        return genotypes[:, keep_snps], snp_ids[keep_snps], keep_snps
    
    def filter_snps_by_maf(self, genotypes: np.ndarray,
                          snp_ids: np.ndarray,
                          min_maf: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter SNPs by Minor Allele Frequency
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            snp_ids: SNP identifiers  
            min_maf: Minimum MAF threshold
            
        Returns:
            Filtered genotypes, snp_ids, and mask of kept SNPs
        """
        maf_values = calculate_maf(genotypes)
        keep_snps = maf_values >= min_maf
        
        n_removed = np.sum(~keep_snps)
        if self.verbose:
            self.logger.info(f"MAF filter: removed {n_removed} SNPs "
                           f"({n_removed/len(snp_ids)*100:.2f}%) with MAF < {min_maf}")
            
        self.qc_stats['snps_removed_maf'] = n_removed
        self.qc_stats['maf_threshold'] = min_maf
        
        return genotypes[:, keep_snps], snp_ids[keep_snps], keep_snps
    
    def filter_snps_by_hwe(self, genotypes: np.ndarray,
                          snp_ids: np.ndarray,
                          hwe_pvalue_threshold: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter SNPs by Hardy-Weinberg Equilibrium
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            snp_ids: SNP identifiers
            hwe_pvalue_threshold: Minimum HWE p-value threshold
            
        Returns:
            Filtered genotypes, snp_ids, and mask of kept SNPs
        """
        hwe_pvalues = calculate_hwe_pvalue(genotypes)
        keep_snps = (hwe_pvalues >= hwe_pvalue_threshold) | np.isnan(hwe_pvalues)
        
        n_removed = np.sum(~keep_snps)
        if self.verbose:
            self.logger.info(f"HWE filter: removed {n_removed} SNPs "
                           f"({n_removed/len(snp_ids)*100:.2f}%) with HWE p-value < {hwe_pvalue_threshold}")
            
        self.qc_stats['snps_removed_hwe'] = n_removed
        self.qc_stats['hwe_threshold'] = hwe_pvalue_threshold
        
        return genotypes[:, keep_snps], snp_ids[keep_snps], keep_snps
    
    def detect_related_samples(self, genotypes: np.ndarray,
                             sample_ids: np.ndarray,
                             kinship_threshold: float = 0.125) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and remove related samples based on kinship coefficient
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            sample_ids: Sample identifiers
            kinship_threshold: Kinship threshold (0.125 = 1st cousins)
            
        Returns:
            Filtered genotypes and sample_ids
        """
        from .population import PopulationStructure
        
        # Calculate kinship matrix
        pop_struct = PopulationStructure()
        kinship_matrix = pop_struct.calculate_kinship_matrix(genotypes)
        
        # Find pairs of related samples
        n_samples = kinship_matrix.shape[0]
        to_remove = set()
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if kinship_matrix[i, j] > kinship_threshold:
                    # Remove sample with lower call rate
                    call_rate_i = calculate_call_rate(genotypes[i:i+1, :], axis=1)[0]
                    call_rate_j = calculate_call_rate(genotypes[j:j+1, :], axis=1)[0]
                    
                    if call_rate_i < call_rate_j:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
        
        keep_samples = np.array([i for i in range(n_samples) if i not in to_remove])
        
        if self.verbose:
            self.logger.info(f"Relatedness filter: removed {len(to_remove)} samples "
                           f"with kinship > {kinship_threshold}")
            
        self.qc_stats['samples_removed_related'] = len(to_remove)
        self.qc_stats['kinship_threshold'] = kinship_threshold
        
        # Ensure sample_ids is a numpy array for boolean indexing
        sample_ids_array = np.array(sample_ids)
        return genotypes[keep_samples, :], sample_ids_array[keep_samples]
    
    def detect_population_outliers(self, genotypes: np.ndarray,
                                 sample_ids: np.ndarray,
                                 n_components: int = 10,
                                 sd_threshold: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect population outliers using PCA
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            sample_ids: Sample identifiers
            n_components: Number of PCs to calculate
            sd_threshold: Standard deviation threshold for outlier detection
            
        Returns:
            Filtered genotypes and sample_ids
        """
        from .population import PopulationStructure
        
        # Calculate PCs
        pop_struct = PopulationStructure()
        pcs, _ = pop_struct.calculate_pca(genotypes, n_components=n_components)
        
        # Detect outliers based on PC values
        outliers = set()
        
        for pc_idx in range(min(3, n_components)):  # Check first 3 PCs
            pc_values = pcs[:, pc_idx]
            mean_pc = np.mean(pc_values)
            std_pc = np.std(pc_values)
            
            outlier_mask = np.abs(pc_values - mean_pc) > sd_threshold * std_pc
            outliers.update(np.where(outlier_mask)[0])
        
        keep_samples = np.array([i for i in range(len(sample_ids)) if i not in outliers])
        
        if self.verbose:
            self.logger.info(f"Population outlier filter: removed {len(outliers)} samples "
                           f"beyond {sd_threshold} SD from mean")
            
        self.qc_stats['samples_removed_outliers'] = len(outliers)
        self.qc_stats['outlier_sd_threshold'] = sd_threshold
        
        # Ensure sample_ids is a numpy array for boolean indexing
        sample_ids_array = np.array(sample_ids)
        return genotypes[keep_samples, :], sample_ids_array[keep_samples]
    
    def filter_phenotype_outliers(self, phenotypes: Union[np.ndarray, pd.DataFrame],
                                 sample_ids: np.ndarray,
                                 sd_threshold: float = 5.0) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        Filter extreme phenotype outliers
        
        Args:
            phenotypes: Phenotype values (can be DataFrame or array)
            sample_ids: Sample identifiers
            sd_threshold: Standard deviation threshold
            
        Returns:
            Filtered phenotypes and sample_ids
        """
        # Handle DataFrame case - use first numeric column for outlier detection
        if isinstance(phenotypes, pd.DataFrame):
            # Find first numeric column
            numeric_cols = phenotypes.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                # No numeric columns, return as-is
                return phenotypes, sample_ids
            
            # Use first numeric column for outlier detection
            pheno_values = phenotypes[numeric_cols[0]].values
        else:
            # Convert to numpy array if it's a pandas Series
            if hasattr(phenotypes, 'values'):
                pheno_values = phenotypes.values
            else:
                pheno_values = np.array(phenotypes)
            
        valid_pheno = ~np.isnan(pheno_values)
        
        if np.sum(valid_pheno) == 0:
            return phenotypes, sample_ids
            
        mean_pheno = np.mean(pheno_values[valid_pheno])
        std_pheno = np.std(pheno_values[valid_pheno])
        
        # Identify outliers
        outlier_mask = np.abs(pheno_values - mean_pheno) > sd_threshold * std_pheno
        outlier_mask = outlier_mask & valid_pheno  # Only consider non-missing values
        
        keep_samples = ~outlier_mask
        
        n_removed = np.sum(outlier_mask)
        if self.verbose:
            self.logger.info(f"Phenotype outlier filter: removed {n_removed} samples "
                           f"beyond {sd_threshold} SD from mean")
            
        self.qc_stats['samples_removed_pheno_outliers'] = n_removed
        self.qc_stats['pheno_outlier_sd_threshold'] = sd_threshold
        
        # Handle pandas DataFrame, Series, or numpy array properly
        if isinstance(phenotypes, pd.DataFrame):
            # DataFrame - filter rows
            filtered_phenotypes = phenotypes.iloc[keep_samples]
        elif hasattr(phenotypes, 'iloc'):
            # pandas Series
            filtered_phenotypes = phenotypes.iloc[keep_samples]
        else:
            # numpy array
            filtered_phenotypes = phenotypes[keep_samples]
        
        # Ensure sample_ids is a numpy array for boolean indexing
        sample_ids_array = np.array(sample_ids)
        return filtered_phenotypes, sample_ids_array[keep_samples]
    
    def comprehensive_qc(self, genotypes: np.ndarray,
                        phenotypes: np.ndarray,
                        sample_ids: np.ndarray,
                        snp_ids: np.ndarray,
                        sample_call_rate: float = 0.95,
                        snp_call_rate: float = 0.95,
                        min_maf: float = 0.01,
                        hwe_threshold: float = 1e-6,
                        kinship_threshold: float = 0.125,
                        population_outlier_sd: float = 6.0,
                        phenotype_outlier_sd: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run comprehensive quality control pipeline
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            phenotypes: Phenotype values
            sample_ids: Sample identifiers
            snp_ids: SNP identifiers
            sample_call_rate: Minimum sample call rate
            snp_call_rate: Minimum SNP call rate  
            min_maf: Minimum minor allele frequency
            hwe_threshold: HWE p-value threshold
            kinship_threshold: Kinship coefficient threshold
            population_outlier_sd: Population outlier SD threshold
            phenotype_outlier_sd: Phenotype outlier SD threshold
            
        Returns:
            Filtered genotypes, phenotypes, sample_ids, snp_ids
        """
        # Validate input data
        validate_input_data(genotypes, phenotypes)
        
        initial_samples = len(sample_ids)
        initial_snps = len(snp_ids)
        
        if self.verbose:
            self.logger.info(f"Starting QC with {initial_samples} samples and {initial_snps} SNPs")
        
        # 1. Filter samples by call rate
        genotypes, sample_ids, sample_mask = self.filter_samples_by_call_rate(
            genotypes, sample_ids, sample_call_rate)
        
        # Handle phenotypes indexing properly
        if hasattr(phenotypes, 'iloc'):
            # pandas Series
            phenotypes = phenotypes.iloc[sample_mask]
        else:
            # numpy array
            phenotypes = phenotypes[sample_mask]
        
        # 2. Filter SNPs by call rate
        genotypes, snp_ids, _ = self.filter_snps_by_call_rate(
            genotypes, snp_ids, snp_call_rate)
        
        # 3. Filter SNPs by MAF
        genotypes, snp_ids, _ = self.filter_snps_by_maf(
            genotypes, snp_ids, min_maf)
        
        # 4. Filter SNPs by HWE
        genotypes, snp_ids, _ = self.filter_snps_by_hwe(
            genotypes, snp_ids, hwe_threshold)
        
        # 5. Filter related samples
        if kinship_threshold > 0:
            original_sample_ids = sample_ids.copy()
            genotypes, sample_ids = self.detect_related_samples(
                genotypes, sample_ids, kinship_threshold)
            # Create mask for keeping samples based on sample IDs
            sample_mask = np.isin(original_sample_ids, sample_ids)
            
            # Handle phenotypes indexing properly
            if hasattr(phenotypes, 'iloc'):
                # pandas Series - need to get the indices that match the kept samples
                keep_indices = np.where(sample_mask)[0]
                phenotypes = phenotypes.iloc[keep_indices]
            else:
                # numpy array
                phenotypes = phenotypes[sample_mask]
        
        # 6. Filter population outliers
        if population_outlier_sd > 0:
            original_sample_ids = sample_ids.copy()
            genotypes, sample_ids = self.detect_population_outliers(
                genotypes, sample_ids, sd_threshold=population_outlier_sd)
            # Create mask for keeping samples based on sample IDs
            sample_mask = np.isin(original_sample_ids, sample_ids)
            
            # Handle phenotypes indexing properly
            if hasattr(phenotypes, 'iloc'):
                # pandas Series - need to get the indices that match the kept samples
                keep_indices = np.where(sample_mask)[0]
                phenotypes = phenotypes.iloc[keep_indices]
            else:
                # numpy array
                phenotypes = phenotypes[sample_mask]
        
        # 7. Filter phenotype outliers
        if phenotype_outlier_sd > 0:
            original_sample_ids = sample_ids.copy()
            phenotypes, sample_ids = self.filter_phenotype_outliers(
                phenotypes, sample_ids, phenotype_outlier_sd)
            # Create mask for keeping samples based on sample IDs
            sample_mask = np.isin(original_sample_ids, sample_ids)
            genotypes = genotypes[sample_mask, :]
        
        final_samples = len(sample_ids)
        final_snps = len(snp_ids)
        
        if self.verbose:
            self.logger.info(f"QC complete: {final_samples} samples ({final_samples/initial_samples*100:.2f}%) "
                           f"and {final_snps} SNPs ({final_snps/initial_snps*100:.2f}%) remaining")
        
        # Store final QC statistics
        self.qc_stats['initial_samples'] = initial_samples
        self.qc_stats['initial_snps'] = initial_snps
        self.qc_stats['final_samples'] = final_samples
        self.qc_stats['final_snps'] = final_snps
        
        return genotypes, phenotypes, sample_ids, snp_ids
    
    def get_qc_report(self) -> pd.DataFrame:
        """
        Generate QC report as DataFrame
        
        Returns:
            DataFrame with QC statistics
        """
        if not self.qc_stats:
            return pd.DataFrame()
            
        report_data = []
        for key, value in self.qc_stats.items():
            report_data.append({'Metric': key, 'Value': value})
            
        return pd.DataFrame(report_data)
    
    def save_qc_report(self, filename: str):
        """
        Save QC report to file
        
        Args:
            filename: Output filename
        """
        report = self.get_qc_report()
        report.to_csv(filename, index=False)
        
        if self.verbose:
            self.logger.info(f"QC report saved to {filename}")
