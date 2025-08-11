"""
Utility functions for PyGWAS
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Union, List
import warnings

def calculate_maf(genotypes: np.ndarray) -> np.ndarray:
    """
    Calculate Minor Allele Frequency (MAF) for each SNP
    
    Args:
        genotypes: Genotype matrix (samples x SNPs), coded as 0, 1, 2, or nan
        
    Returns:
        Array of MAF values for each SNP
    """
    # Count alleles (0=AA, 1=AB, 2=BB)
    n_samples = np.sum(~np.isnan(genotypes), axis=0)
    allele_counts = np.nansum(genotypes, axis=0)
    total_alleles = 2 * n_samples
    
    # Calculate allele frequency
    freq = allele_counts / total_alleles
    
    # MAF is the minimum of freq and 1-freq
    maf = np.minimum(freq, 1 - freq)
    
    return maf

def calculate_hwe_pvalue(genotypes: np.ndarray) -> np.ndarray:
    """
    Calculate Hardy-Weinberg Equilibrium p-values for each SNP
    
    Args:
        genotypes: Genotype matrix (samples x SNPs)
        
    Returns:
        Array of HWE p-values
    """
    n_snps = genotypes.shape[1]
    hwe_pvalues = np.zeros(n_snps)
    
    for i in range(n_snps):
        geno = genotypes[:, i]
        geno = geno[~np.isnan(geno)]
        
        if len(geno) == 0:
            hwe_pvalues[i] = np.nan
            continue
            
        # Count genotypes
        n_aa = np.sum(geno == 0)
        n_ab = np.sum(geno == 1) 
        n_bb = np.sum(geno == 2)
        n_total = n_aa + n_ab + n_bb
        
        if n_total == 0:
            hwe_pvalues[i] = np.nan
            continue
            
        # Calculate expected frequencies under HWE
        p = (2 * n_aa + n_ab) / (2 * n_total)  # Allele frequency
        q = 1 - p
        
        exp_aa = n_total * p * p
        exp_ab = n_total * 2 * p * q
        exp_bb = n_total * q * q
        
        # Chi-square test
        if exp_aa > 0 and exp_ab > 0 and exp_bb > 0:
            chi2 = ((n_aa - exp_aa)**2 / exp_aa + 
                   (n_ab - exp_ab)**2 / exp_ab + 
                   (n_bb - exp_bb)**2 / exp_bb)
            hwe_pvalues[i] = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            hwe_pvalues[i] = np.nan
            
    return hwe_pvalues

def calculate_call_rate(genotypes: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate call rate (proportion of non-missing genotypes)
    
    Args:
        genotypes: Genotype matrix
        axis: 0 for sample call rates, 1 for SNP call rates
        
    Returns:
        Array of call rates
    """
    missing = np.isnan(genotypes)
    call_rate = 1 - np.mean(missing, axis=axis)
    return call_rate

def standardize_genotypes(genotypes: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Standardize genotype matrix
    
    Args:
        genotypes: Genotype matrix (samples x SNPs)
        method: 'standard' or 'unit_variance'
        
    Returns:
        Standardized genotype matrix
    """
    # Handle missing values by replacing with mean
    standardized = genotypes.copy()
    
    for i in range(genotypes.shape[1]):
        col = genotypes[:, i]
        valid_mask = ~np.isnan(col)
        
        if np.sum(valid_mask) == 0:
            continue
            
        mean_val = np.mean(col[valid_mask])
        standardized[~valid_mask, i] = mean_val
        
        if method == 'standard':
            std_val = np.std(col[valid_mask])
            if std_val > 0:
                standardized[:, i] = (standardized[:, i] - mean_val) / std_val
        elif method == 'unit_variance':
            # For dosage data: Var(X) = 2*p*(1-p)
            p = mean_val / 2
            var_val = 2 * p * (1 - p)
            if var_val > 0:
                standardized[:, i] = (standardized[:, i] - mean_val) / np.sqrt(var_val)
                
    return standardized

def genomic_control_lambda(pvalues: np.ndarray) -> float:
    """
    Calculate genomic control lambda (inflation factor)
    
    Args:
        pvalues: Array of association p-values
        
    Returns:
        Lambda value
    """
    # Remove NaN values
    valid_pvals = pvalues[~np.isnan(pvalues)]
    
    if len(valid_pvals) == 0:
        return np.nan
        
    # Convert to chi-square statistics
    chi2_stats = stats.chi2.ppf(1 - valid_pvals, df=1)
    
    # Calculate lambda as median(chi2) / median(chi2_expected)
    observed_median = np.median(chi2_stats)
    expected_median = stats.chi2.ppf(0.5, df=1)
    
    lambda_gc = observed_median / expected_median
    
    return lambda_gc

def correct_pvalues_gc(pvalues: np.ndarray, lambda_gc: float) -> np.ndarray:
    """
    Apply genomic control correction to p-values
    
    Args:
        pvalues: Original p-values
        lambda_gc: Genomic control lambda
        
    Returns:
        Corrected p-values
    """
    if lambda_gc <= 0:
        return pvalues
        
    # Convert to chi-square, correct, and convert back
    chi2_stats = stats.chi2.ppf(1 - pvalues, df=1)
    corrected_chi2 = chi2_stats / lambda_gc
    corrected_pvals = 1 - stats.chi2.cdf(corrected_chi2, df=1)
    
    return corrected_pvals

def multiple_testing_correction(pvalues: np.ndarray, method: str = 'fdr_bh') -> np.ndarray:
    """
    Apply multiple testing correction
    
    Args:
        pvalues: Array of p-values
        method: 'fdr_bh' (Benjamini-Hochberg) or 'bonferroni'
        
    Returns:
        Corrected p-values
    """
    valid_mask = ~np.isnan(pvalues)
    valid_pvals = pvalues[valid_mask]
    n_tests = len(valid_pvals)
    
    corrected = np.full_like(pvalues, np.nan)
    
    if n_tests == 0:
        return corrected
        
    if method == 'bonferroni':
        corrected[valid_mask] = np.minimum(valid_pvals * n_tests, 1.0)
    elif method == 'fdr_bh':
        # Benjamini-Hochberg procedure
        sorted_indices = np.argsort(valid_pvals)
        sorted_pvals = valid_pvals[sorted_indices]
        
        # Calculate corrected p-values
        corrected_sorted = np.zeros(n_tests)
        for i in range(n_tests - 1, -1, -1):
            rank = i + 1
            bh_val = sorted_pvals[i] * n_tests / rank
            if i == n_tests - 1:
                corrected_sorted[i] = min(bh_val, 1.0)
            else:
                corrected_sorted[i] = min(bh_val, corrected_sorted[i + 1])
                
        # Map back to original order
        corrected_unsorted = np.zeros(n_tests)
        corrected_unsorted[sorted_indices] = corrected_sorted
        corrected[valid_mask] = corrected_unsorted
        
    return corrected

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared value
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R-squared value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
        
    r2 = 1 - (ss_res / ss_tot)
    return r2

def load_genetic_map(map_file: str) -> pd.DataFrame:
    """
    Load genetic map file (for LD calculations)
    
    Args:
        map_file: Path to genetic map file
        
    Returns:
        DataFrame with genetic positions
    """
    try:
        genetic_map = pd.read_csv(map_file, sep='\t', 
                                 names=['chr', 'pos', 'rate', 'cm'])
        return genetic_map
    except:
        warnings.warn(f"Could not load genetic map from {map_file}")
        return pd.DataFrame()

def validate_input_data(genotypes: np.ndarray, phenotypes: np.ndarray, 
                       covariates: Optional[np.ndarray] = None) -> bool:
    """
    Validate input data for GWAS analysis
    
    Args:
        genotypes: Genotype matrix
        phenotypes: Phenotype vector
        covariates: Covariate matrix (optional)
        
    Returns:
        True if data is valid
    """
    # Check dimensions
    if genotypes.shape[0] != len(phenotypes):
        raise ValueError("Number of samples in genotypes and phenotypes must match")
        
    if covariates is not None and covariates.shape[0] != len(phenotypes):
        raise ValueError("Number of samples in covariates and phenotypes must match")
        
    # Check for valid genotype values
    valid_geno = np.isin(genotypes[~np.isnan(genotypes)], [0, 1, 2])
    if not np.all(valid_geno):
        raise ValueError("Genotypes must be coded as 0, 1, 2, or NaN")
        
    # Check for valid phenotype values
    if np.all(np.isnan(phenotypes)):
        raise ValueError("All phenotype values are missing")
        
    return True
