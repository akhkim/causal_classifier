import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve, inv, LinAlgError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union, Dict, List
import warnings
from numba import njit
import multiprocessing as mp
from tqdm import tqdm

from .utils import (
    standardize_genotypes, genomic_control_lambda, 
    correct_pvalues_gc, multiple_testing_correction,
    calculate_r_squared, validate_input_data
)

class AssociationTest:
    """
    Association testing methods for GWAS
    """
    
    def __init__(self, n_jobs: int = -1, verbose: bool = True):
        """
        Initialize AssociationTest object
        
        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to print progress messages
        """
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.verbose = verbose
        self.results = None
        
    def linear_regression_test(self, genotypes: np.ndarray,
                              phenotypes: np.ndarray,
                              covariates: Optional[np.ndarray] = None,
                              standardize_geno: bool = True,
                              variant_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform linear regression association test for quantitative traits
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            phenotypes: Phenotype values (continuous)
            covariates: Covariate matrix (optional)
            standardize_geno: Whether to standardize genotypes
            
        Returns:
            DataFrame with association results
        """
        if self.verbose:
            print("Running linear regression association test...")
            
        # Validate input - reshape phenotypes to 2D for validation if needed
        phenotypes_for_validation = phenotypes.reshape(-1, 1) if phenotypes.ndim == 1 else phenotypes
        validate_input_data(genotypes, phenotypes_for_validation, covariates)
        
        n_samples, n_snps = genotypes.shape
        
        # Remove samples with missing phenotypes
        valid_pheno = ~np.isnan(phenotypes)
        genotypes = genotypes[valid_pheno, :]
        phenotypes = phenotypes[valid_pheno]
        if covariates is not None:
            covariates = covariates[valid_pheno, :]
            
        n_valid = len(phenotypes)
        if self.verbose:
            print(f"Testing {n_snps} SNPs in {n_valid} samples")
            
        # Standardize genotypes if requested
        if standardize_geno:
            genotypes = standardize_genotypes(genotypes)
            
        # Prepare design matrix
        if covariates is not None:
            # Add intercept to covariates
            X_cov = np.column_stack([np.ones(n_valid), covariates])
        else:
            X_cov = np.ones((n_valid, 1))
            
        # Pre-compute covariate effects
        try:
            XtX_cov_inv = inv(X_cov.T @ X_cov)
            beta_cov = XtX_cov_inv @ X_cov.T @ phenotypes
            residuals = phenotypes - X_cov @ beta_cov
            sigma2_null = np.var(residuals)
        except LinAlgError:
            warnings.warn("Covariate matrix is singular, using pseudoinverse")
            beta_cov = np.linalg.pinv(X_cov) @ phenotypes
            residuals = phenotypes - X_cov @ beta_cov
            sigma2_null = np.var(residuals)
            
        # Initialize results
        results = {
            'CHR': [],
            'POS': [],
            'SNP': [],
            'A1': [],
            'A2': [],
            'N': [],
            'BETA': [],
            'SE': [],
            'T_STAT': [],
            'P': [],
            'R2': []
        }
        
        # Test each SNP
        for snp_idx in tqdm(range(n_snps), disable=not self.verbose, desc="Testing SNPs"):
            geno = genotypes[:, snp_idx]
            
            # Skip SNPs with all missing data or no variation
            valid_geno = ~np.isnan(geno)
            if np.sum(valid_geno) < 10 or np.var(geno[valid_geno]) == 0:
                # Add NaN results with proper variant info if available
                if variant_info is not None and snp_idx < len(variant_info):
                    var_row = variant_info.iloc[snp_idx]
                    results['CHR'].append(var_row.get('CHR', 'NA'))
                    results['POS'].append(var_row.get('POS', 'NA'))
                    results['SNP'].append(var_row.get('SNP', f'SNP_{snp_idx+1}'))
                    results['A1'].append(var_row.get('A1', 'NA'))
                    results['A2'].append(var_row.get('A2', 'NA'))
                else:
                    results['CHR'].append('NA')
                    results['POS'].append('NA')
                    results['SNP'].append(f'SNP_{snp_idx+1}')
                    results['A1'].append('NA')
                    results['A2'].append('NA')
                
                # Add NaN for statistics
                for key in ['N', 'BETA', 'SE', 'T_STAT', 'P', 'R2']:
                    results[key].append(np.nan)
                continue
                
            # Subset to valid genotypes
            y_valid = phenotypes[valid_geno]
            X_cov_valid = X_cov[valid_geno, :]
            g_valid = geno[valid_geno]
            
            # Design matrix with genotype
            X_full = np.column_stack([X_cov_valid, g_valid])
            
            try:
                # Fit full model
                XtX_inv = inv(X_full.T @ X_full)
                beta_full = XtX_inv @ X_full.T @ y_valid
                
                # Extract genotype effect
                beta_geno = beta_full[-1]
                se_geno = np.sqrt(XtX_inv[-1, -1] * sigma2_null)
                
                # T-test
                t_stat = beta_geno / se_geno
                df = len(y_valid) - X_full.shape[1]
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
                
                # R-squared
                y_pred = X_full @ beta_full
                r2 = calculate_r_squared(y_valid, y_pred)
                
            except (LinAlgError, ValueError):
                # Fallback to sklearn for problematic cases
                try:
                    if X_cov_valid.shape[1] > 1:  # Have covariates
                        # Fit null model first
                        lr_null = LinearRegression()
                        lr_null.fit(X_cov_valid, y_valid)
                        y_resid = y_valid - lr_null.predict(X_cov_valid)
                        
                        # Test genotype on residuals
                        lr_geno = LinearRegression()
                        lr_geno.fit(g_valid.reshape(-1, 1), y_resid)
                        beta_geno = lr_geno.coef_[0]
                        
                        # Calculate standard error
                        y_pred_resid = lr_geno.predict(g_valid.reshape(-1, 1))
                        mse = np.mean((y_resid - y_pred_resid)**2)
                        se_geno = np.sqrt(mse / np.sum((g_valid - np.mean(g_valid))**2))
                        
                    else:  # No covariates
                        lr = LinearRegression()
                        lr.fit(g_valid.reshape(-1, 1), y_valid)
                        beta_geno = lr.coef_[0]
                        
                        # Calculate standard error
                        y_pred = lr.predict(g_valid.reshape(-1, 1))
                        mse = np.mean((y_valid - y_pred)**2)
                        se_geno = np.sqrt(mse / np.sum((g_valid - np.mean(g_valid))**2))
                        
                    t_stat = beta_geno / se_geno
                    df = len(y_valid) - 2  # Approximate
                    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
                    r2 = calculate_r_squared(y_valid, y_pred)
                    
                except:
                    beta_geno = se_geno = t_stat = p_value = r2 = np.nan
                    
            # Store results with proper variant info if available
            if variant_info is not None and snp_idx < len(variant_info):
                var_row = variant_info.iloc[snp_idx]
                results['CHR'].append(var_row.get('CHR', 'NA'))
                results['POS'].append(var_row.get('POS', 'NA'))
                results['SNP'].append(var_row.get('SNP', f'SNP_{snp_idx+1}'))
                results['A1'].append(var_row.get('A1', 'NA'))
                results['A2'].append(var_row.get('A2', 'NA'))
            else:
                results['CHR'].append('NA')
                results['POS'].append('NA')
                results['SNP'].append(f'SNP_{snp_idx+1}')
                results['A1'].append('NA')
                results['A2'].append('NA')
            results['N'].append(len(y_valid))
            results['BETA'].append(beta_geno)
            results['SE'].append(se_geno)
            results['T_STAT'].append(t_stat)
            results['P'].append(p_value)
            results['R2'].append(r2)
            
        return pd.DataFrame(results)
    
    def logistic_regression_test(self, genotypes: np.ndarray,
                               phenotypes: np.ndarray,
                               covariates: Optional[np.ndarray] = None,
                               standardize_geno: bool = True,
                               variant_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform logistic regression association test for binary traits
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            phenotypes: Binary phenotype values (0/1)
            covariates: Covariate matrix (optional)
            standardize_geno: Whether to standardize genotypes
            
        Returns:
            DataFrame with association results
        """
        if self.verbose:
            print("Running logistic regression association test...")
            
        # Validate input - reshape phenotypes to 2D for validation if needed
        phenotypes_for_validation = phenotypes.reshape(-1, 1) if phenotypes.ndim == 1 else phenotypes
        validate_input_data(genotypes, phenotypes_for_validation, covariates)
        
        # Check if phenotype is binary
        unique_pheno = np.unique(phenotypes[~np.isnan(phenotypes)])
        if len(unique_pheno) > 2 or not all(p in [0, 1] for p in unique_pheno):
            warnings.warn("Phenotype should be binary (0/1) for logistic regression")
            
        n_samples, n_snps = genotypes.shape
        
        # Remove samples with missing phenotypes
        valid_pheno = ~np.isnan(phenotypes)
        genotypes = genotypes[valid_pheno, :]
        phenotypes = phenotypes[valid_pheno]
        if covariates is not None:
            covariates = covariates[valid_pheno, :]
            
        n_valid = len(phenotypes)
        if self.verbose:
            print(f"Testing {n_snps} SNPs in {n_valid} samples")
            
        # Standardize genotypes if requested
        if standardize_geno:
            genotypes = standardize_genotypes(genotypes)
            
        # Initialize results
        results = {
            'CHR': [],
            'POS': [],
            'SNP': [],
            'A1': [],
            'A2': [],
            'N': [],
            'OR': [],
            'SE': [],
            'Z_STAT': [],
            'P': [],
            'N_CASE': [],
            'N_CTRL': []
        }
        
        # Count cases and controls
        n_cases = np.sum(phenotypes == 1)
        n_controls = np.sum(phenotypes == 0)
        
        # Test each SNP
        for snp_idx in tqdm(range(n_snps), disable=not self.verbose, desc="Testing SNPs"):
            geno = genotypes[:, snp_idx]
            
            # Skip SNPs with all missing data or no variation
            valid_geno = ~np.isnan(geno)
            if np.sum(valid_geno) < 10 or np.var(geno[valid_geno]) == 0:
                # Add NaN results
                for key in results.keys():
                    if key in ['CHR', 'POS', 'SNP', 'A1', 'A2']:
                        results[key].append(f'SNP_{snp_idx+1}' if key == 'SNP' else 'NA')
                    elif key in ['N_CASE', 'N_CTRL']:
                        results[key].append(0)
                    else:
                        results[key].append(np.nan)
                continue
                
            # Subset to valid genotypes
            y_valid = phenotypes[valid_geno]
            g_valid = geno[valid_geno]
            
            # Prepare design matrix
            if covariates is not None:
                X_cov_valid = covariates[valid_geno, :]
                X_full = np.column_stack([X_cov_valid, g_valid])
            else:
                X_full = g_valid.reshape(-1, 1)
                
            try:
                # Fit logistic regression
                lr = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
                lr.fit(X_full, y_valid)
                
                # Extract genotype effect (last coefficient)
                beta_geno = lr.coef_[0, -1]
                or_geno = np.exp(beta_geno)
                
                # Calculate standard error using Fisher Information
                # This is an approximation - for exact SE would need Hessian
                y_pred_prob = lr.predict_proba(X_full)[:, 1]
                weights = y_pred_prob * (1 - y_pred_prob)
                
                # Weighted least squares approximation for SE
                X_weighted = X_full * np.sqrt(weights).reshape(-1, 1)
                try:
                    fisher_info = X_weighted.T @ X_weighted
                    cov_matrix = inv(fisher_info)
                    se_geno = np.sqrt(cov_matrix[-1, -1])
                except:
                    # Fallback SE calculation
                    se_geno = np.abs(beta_geno) / 1.96  # Rough approximation
                    
                # Z-test
                z_stat = beta_geno / se_geno
                p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
                
            except (ValueError, LinAlgError):
                # Fallback for convergence issues
                try:
                    # Simple 2x2 contingency table test for genotype effect only
                    if covariates is None:
                        # Dichotomize genotype for simplicity
                        g_binary = (g_valid > np.median(g_valid)).astype(int)
                        
                        # 2x2 table
                        table = np.zeros((2, 2))
                        table[0, 0] = np.sum((y_valid == 0) & (g_binary == 0))  # Controls, no variant
                        table[0, 1] = np.sum((y_valid == 0) & (g_binary == 1))  # Controls, variant
                        table[1, 0] = np.sum((y_valid == 1) & (g_binary == 0))  # Cases, no variant
                        table[1, 1] = np.sum((y_valid == 1) & (g_binary == 1))  # Cases, variant
                        
                        # Calculate OR and p-value
                        if table[0, 1] > 0 and table[1, 0] > 0:
                            or_geno = (table[1, 1] * table[0, 0]) / (table[1, 0] * table[0, 1])
                            beta_geno = np.log(or_geno)
                            
                            # Chi-square test
                            chi2, p_value, _, _ = stats.chi2_contingency(table)
                            
                            # Approximate SE from OR confidence interval
                            se_geno = np.sqrt(1/table[0, 0] + 1/table[0, 1] + 1/table[1, 0] + 1/table[1, 1])
                            z_stat = beta_geno / se_geno
                        else:
                            or_geno = beta_geno = se_geno = z_stat = p_value = np.nan
                    else:
                        or_geno = beta_geno = se_geno = z_stat = p_value = np.nan
                        
                except:
                    or_geno = beta_geno = se_geno = z_stat = p_value = np.nan
                    
            # Store results with proper variant info if available
            if variant_info is not None and snp_idx < len(variant_info):
                var_row = variant_info.iloc[snp_idx]
                results['CHR'].append(var_row.get('CHR', 'NA'))
                results['POS'].append(var_row.get('POS', 'NA'))
                results['SNP'].append(var_row.get('SNP', f'SNP_{snp_idx+1}'))
                results['A1'].append(var_row.get('A1', 'NA'))
                results['A2'].append(var_row.get('A2', 'NA'))
            else:
                results['CHR'].append('NA')
                results['POS'].append('NA')
                results['SNP'].append(f'SNP_{snp_idx+1}')
                results['A1'].append('NA')
                results['A2'].append('NA')
            results['N'].append(len(y_valid))
            results['OR'].append(or_geno)
            results['SE'].append(se_geno)
            results['Z_STAT'].append(z_stat)
            results['P'].append(p_value)
            results['N_CASE'].append(n_cases)
            results['N_CTRL'].append(n_controls)
            
        return pd.DataFrame(results)
    
    def mixed_linear_model_test(self, genotypes: np.ndarray,
                              phenotypes: np.ndarray,
                              kinship_matrix: np.ndarray,
                              covariates: Optional[np.ndarray] = None,
                              variant_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Mixed Linear Model association test accounting for population structure
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            phenotypes: Phenotype values
            kinship_matrix: Kinship/relationship matrix
            covariates: Covariate matrix (optional)
            
        Returns:
            DataFrame with association results
        """
        if self.verbose:
            print("Running Mixed Linear Model association test...")
            
        # This is a simplified implementation
        # Full implementation would use REML for variance component estimation
        
        n_samples, n_snps = genotypes.shape
        
        # Remove samples with missing phenotypes
        valid_pheno = ~np.isnan(phenotypes)
        genotypes = genotypes[valid_pheno, :]
        phenotypes = phenotypes[valid_pheno]
        kinship_matrix = kinship_matrix[np.ix_(valid_pheno, valid_pheno)]
        if covariates is not None:
            covariates = covariates[valid_pheno, :]
            
        n_valid = len(phenotypes)
        
        # Eigendecomposition of kinship matrix for efficiency
        try:
            eigenvals, eigenvecs = np.linalg.eigh(kinship_matrix)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Regularize
        except:
            warnings.warn("Kinship matrix eigendecomposition failed, using identity")
            eigenvals = np.ones(n_valid)
            eigenvecs = np.eye(n_valid)
            
        # Transform data
        y_transformed = eigenvecs.T @ phenotypes
        
        if covariates is not None:
            X_cov = np.column_stack([np.ones(n_valid), covariates])
        else:
            X_cov = np.ones((n_valid, 1))
            
        X_cov_transformed = eigenvecs.T @ X_cov
        
        # Estimate variance components (simplified - should use REML)
        # Here we use a crude approximation
        h2 = 0.5  # Assume heritability of 0.5
        sigma2_g = h2
        sigma2_e = 1 - h2
        
        # Weight matrix
        weights = 1 / (sigma2_g * eigenvals + sigma2_e)
        W = np.diag(weights)
        
        # Initialize results
        results = {
            'CHR': [],
            'POS': [],
            'SNP': [],
            'A1': [],
            'A2': [],
            'N': [],
            'BETA': [],
            'SE': [],
            'T_STAT': [],
            'P': []
        }
        
        # Test each SNP
        for snp_idx in tqdm(range(n_snps), disable=not self.verbose, desc="Testing SNPs"):
            geno = genotypes[:, snp_idx]
            
            # Skip SNPs with missing data or no variation
            valid_geno = ~np.isnan(geno)
            if np.sum(valid_geno) < 10 or np.var(geno[valid_geno]) == 0:
                # Add NaN results
                for key in results.keys():
                    if key in ['CHR', 'POS', 'SNP', 'A1', 'A2']:
                        results[key].append(f'SNP_{snp_idx+1}' if key == 'SNP' else 'NA')
                    else:
                        results[key].append(np.nan)
                continue
                
            # Transform genotype
            g_transformed = eigenvecs.T @ geno
            
            # Design matrix
            X_full_transformed = np.column_stack([X_cov_transformed, g_transformed])
            
            try:
                # Weighted least squares
                XtWX = X_full_transformed.T @ W @ X_full_transformed
                XtWy = X_full_transformed.T @ W @ y_transformed
                
                beta = solve(XtWX, XtWy)
                
                # Extract genotype effect
                beta_geno = beta[-1]
                se_geno = np.sqrt(inv(XtWX)[-1, -1])
                
                # T-test
                t_stat = beta_geno / se_geno
                df = n_valid - X_full_transformed.shape[1]
                p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
                
            except (LinAlgError, ValueError):
                beta_geno = se_geno = t_stat = p_value = np.nan
                
            # Store results with proper variant info if available
            if variant_info is not None and snp_idx < len(variant_info):
                var_row = variant_info.iloc[snp_idx]
                results['CHR'].append(var_row.get('CHR', 'NA'))
                results['POS'].append(var_row.get('POS', 'NA'))
                results['SNP'].append(var_row.get('SNP', f'SNP_{snp_idx+1}'))
                results['A1'].append(var_row.get('A1', 'NA'))
                results['A2'].append(var_row.get('A2', 'NA'))
            else:
                results['CHR'].append('NA')
                results['POS'].append('NA')
                results['SNP'].append(f'SNP_{snp_idx+1}')
                results['A1'].append('NA')
                results['A2'].append('NA')
            results['N'].append(n_valid)
            results['BETA'].append(beta_geno)
            results['SE'].append(se_geno)
            results['T_STAT'].append(t_stat)
            results['P'].append(p_value)
            
        return pd.DataFrame(results)
    
    def run_association(self, genotypes: np.ndarray,
                       phenotypes: np.ndarray,
                       trait_type: str = 'quantitative',
                       covariates: Optional[np.ndarray] = None,
                       kinship_matrix: Optional[np.ndarray] = None,
                       test_method: str = 'auto',
                       variant_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run association test with automatic method selection
        
        Args:
            genotypes: Genotype matrix (samples x SNPs)
            phenotypes: Phenotype values
            trait_type: 'quantitative' or 'binary'
            covariates: Covariate matrix (optional)
            kinship_matrix: Kinship matrix for MLM (optional)
            test_method: 'auto', 'linear', 'logistic', or 'mlm'
            
        Returns:
            DataFrame with association results
        """
        # Auto-detect trait type if not specified
        if trait_type == 'auto':
            unique_vals = np.unique(phenotypes[~np.isnan(phenotypes)])
            if len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals):
                trait_type = 'binary'
            else:
                trait_type = 'quantitative'
                
        # Auto-select test method
        if test_method == 'auto':
            if kinship_matrix is not None:
                test_method = 'mlm'
            elif trait_type == 'binary':
                test_method = 'logistic'
            else:
                test_method = 'linear'
                
        # Run appropriate test
        if test_method == 'linear':
            results = self.linear_regression_test(genotypes, phenotypes, covariates, variant_info=variant_info)
        elif test_method == 'logistic':
            results = self.logistic_regression_test(genotypes, phenotypes, covariates, variant_info=variant_info)
        elif test_method == 'mlm':
            if kinship_matrix is None:
                raise ValueError("Kinship matrix required for MLM test")
            results = self.mixed_linear_model_test(genotypes, phenotypes, kinship_matrix, covariates, variant_info=variant_info)
        else:
            raise ValueError(f"Unknown test method: {test_method}")
            
        # Apply genomic control
        if 'P' in results.columns:
            pvalues = results['P'].values
            lambda_gc = genomic_control_lambda(pvalues)
            
            if self.verbose:
                print(f"Genomic control lambda: {lambda_gc:.4f}")
                
            if lambda_gc > 1.1:  # Apply correction if inflation detected
                results['P_GC'] = correct_pvalues_gc(pvalues, lambda_gc)
                results['LAMBDA_GC'] = lambda_gc
                
        # Apply multiple testing correction
        if 'P' in results.columns:
            p_col = 'P_GC' if 'P_GC' in results.columns else 'P'
            results['P_FDR'] = multiple_testing_correction(results[p_col].values, 'fdr_bh')
            results['P_BONF'] = multiple_testing_correction(results[p_col].values, 'bonferroni')
        
        # Filter out incomplete results
        results = self.filter_incomplete_results(results)
            
        self.results = results
        
        if self.verbose:
            n_sig_nominal = np.sum(results['P'] < 0.05)
            n_sig_bonf = np.sum(results['P_BONF'] < 0.05) if 'P_BONF' in results.columns else 0
            n_sig_fdr = np.sum(results['P_FDR'] < 0.05) if 'P_FDR' in results.columns else 0
            
            print(f"Association test complete:")
            print(f"  Nominally significant (P < 0.05): {n_sig_nominal}")
            print(f"  Bonferroni significant: {n_sig_bonf}")
            print(f"  FDR significant: {n_sig_fdr}")
            
        return results
    
    def filter_incomplete_results(self, results: pd.DataFrame, 
                                 core_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter out rows with missing values in core columns
        
        Args:
            results: Association results DataFrame
            core_columns: List of columns that must have valid values
                         Default: ['CHR', 'POS', 'SNP', 'BETA', 'SE', 'P']
        
        Returns:
            Filtered DataFrame with complete rows only
        """
        if core_columns is None:
            core_columns = ['CHR', 'POS', 'SNP', 'BETA', 'SE', 'P']
        
        # Count initial rows
        initial_count = len(results)
        
        # Filter for valid values in core columns
        mask = pd.Series(True, index=results.index)
        
        for col in core_columns:
            if col in results.columns:
                # Remove rows with NA, 'NA', empty strings, or NaN
                col_mask = (
                    results[col].notna() & 
                    (results[col] != 'NA') & 
                    (results[col] != '') &
                    (results[col] != 'nan')
                )
                mask = mask & col_mask
        
        # Apply filter
        filtered_results = results[mask].copy()
        
        # Report filtering
        filtered_count = len(filtered_results)
        removed_count = initial_count - filtered_count
        
        if self.verbose:
            print(f"Filtering results: {initial_count} total, {removed_count} incomplete rows removed, {filtered_count} complete rows retained")
            if removed_count > 0:
                print(f"  Removed {removed_count/initial_count*100:.1f}% of results due to missing core data")
        
        return filtered_results
    
    def save_results(self, filename: str, format: str = 'csv'):
        """
        Save association results to file
        
        Args:
            filename: Output filename
            format: File format ('csv' or 'tsv')
        """
        if self.results is None:
            raise ValueError("No results to save. Run association test first.")
            
        if format == 'csv':
            self.results.to_csv(filename, index=False)
        elif format == 'tsv':
            self.results.to_csv(filename, sep='\t', index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        if self.verbose:
            print(f"Results saved to {filename}")
