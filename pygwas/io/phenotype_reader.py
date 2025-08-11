"""
Phenotype file reader for PyGWAS
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
import warnings

class PhenotypeReader:
    """
    Phenotype file reader supporting multiple formats
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize phenotype reader
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
    def read_phenotype_file(self, pheno_file: str,
                          sample_id_col: Union[str, int] = 'IID',
                          phenotype_col: Union[str, int] = 'PHENOTYPE',
                          delimiter: Optional[str] = None,
                          missing_values: List[str] = ['-9', 'NA', 'nan', '.']) -> pd.DataFrame:
        """
        Read phenotype file in various formats
        
        Args:
            pheno_file: Path to phenotype file
            sample_id_col: Column name or index for sample IDs
            phenotype_col: Column name or index for phenotype values
            delimiter: Column delimiter (auto-detected if None)
            missing_values: Values to treat as missing
            
        Returns:
            DataFrame with sample IDs and phenotype values
        """
        if self.verbose:
            print(f"Reading phenotype file: {pheno_file}")
            
        # Try to determine file format and read
        try:
            # Auto-detect delimiter if not specified
            if delimiter is None:
                # Read first few lines to detect delimiter
                with open(pheno_file, 'r') as f:
                    first_line = f.readline().strip()
                    
                if '\\t' in first_line:
                    delimiter = '\\t'
                elif ',' in first_line:
                    delimiter = ','
                else:
                    delimiter = '\\s+'  # Multiple whitespace
                    
            # Read the file
            if delimiter == '\\s+':
                df = pd.read_csv(pheno_file, sep=delimiter, engine='python')
            else:
                df = pd.read_csv(pheno_file, sep=delimiter)
                
        except Exception as e:
            raise ValueError(f"Error reading phenotype file: {e}")
            
        # Handle column selection
        if isinstance(sample_id_col, int):
            if sample_id_col >= len(df.columns):
                raise ValueError(f"Sample ID column index {sample_id_col} out of range")
            sample_id_col = df.columns[sample_id_col]
            
        if isinstance(phenotype_col, int):
            if phenotype_col >= len(df.columns):
                raise ValueError(f"Phenotype column index {phenotype_col} out of range")
            phenotype_col = df.columns[phenotype_col]
            
        # Check if columns exist
        if sample_id_col not in df.columns:
            raise ValueError(f"Sample ID column '{sample_id_col}' not found in file")
            
        if phenotype_col not in df.columns:
            raise ValueError(f"Phenotype column '{phenotype_col}' not found in file")
            
        # Extract relevant columns
        pheno_df = df[[sample_id_col, phenotype_col]].copy()
        pheno_df.columns = ['IID', 'PHENOTYPE']
        
        # Handle missing values
        for missing_val in missing_values:
            pheno_df['PHENOTYPE'] = pheno_df['PHENOTYPE'].replace(missing_val, np.nan)
            
        # Convert phenotype to numeric
        try:
            pheno_df['PHENOTYPE'] = pd.to_numeric(pheno_df['PHENOTYPE'], errors='coerce')
        except:
            warnings.warn("Could not convert all phenotype values to numeric")
            
        # Remove completely missing phenotypes
        initial_count = len(pheno_df)
        pheno_df = pheno_df.dropna(subset=['PHENOTYPE'])
        final_count = len(pheno_df)
        
        if self.verbose:
            print(f"Loaded {final_count} samples with phenotype data")
            if initial_count > final_count:
                print(f"Removed {initial_count - final_count} samples with missing phenotypes")
                
        return pheno_df
    
    def read_plink_phenotype(self, fam_file: str, 
                            phenotype_col: str = 'PHENOTYPE') -> pd.DataFrame:
        """
        Read phenotype from PLINK .fam file
        
        Args:
            fam_file: Path to .fam file
            phenotype_col: Which column contains phenotype (PHENOTYPE or SEX)
            
        Returns:
            DataFrame with sample IDs and phenotype values
        """
        if self.verbose:
            print(f"Reading phenotype from PLINK .fam file: {fam_file}")
            
        # Read .fam file
        fam_columns = ['FID', 'IID', 'FATHER', 'MOTHER', 'SEX', 'PHENOTYPE']
        
        try:
            fam_df = pd.read_csv(fam_file, sep='\\s+', header=None, 
                               names=fam_columns, engine='python')
        except:
            fam_df = pd.read_csv(fam_file, sep=None, header=None,
                               names=fam_columns, engine='python')
            
        # Extract phenotype column
        if phenotype_col not in fam_df.columns:
            raise ValueError(f"Column '{phenotype_col}' not found in .fam file")
            
        pheno_df = fam_df[['IID', phenotype_col]].copy()
        pheno_df.columns = ['IID', 'PHENOTYPE']
        
        # Handle PLINK missing values
        pheno_df['PHENOTYPE'] = pheno_df['PHENOTYPE'].replace(['-9', '0'], np.nan)
        pheno_df['PHENOTYPE'] = pd.to_numeric(pheno_df['PHENOTYPE'], errors='coerce')
        
        # Remove missing
        initial_count = len(pheno_df)
        pheno_df = pheno_df.dropna(subset=['PHENOTYPE'])
        final_count = len(pheno_df)
        
        if self.verbose:
            print(f"Loaded {final_count} samples with phenotype data from .fam file")
            if initial_count > final_count:
                print(f"Removed {initial_count - final_count} samples with missing phenotypes")
                
        return pheno_df
    
    def read_covariate_file(self, covar_file: str,
                           sample_id_col: Union[str, int] = 'IID',
                           exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read covariate file
        
        Args:
            covar_file: Path to covariate file
            sample_id_col: Column name or index for sample IDs
            exclude_cols: Columns to exclude from covariates
            
        Returns:
            DataFrame with sample IDs and covariate values
        """
        if self.verbose:
            print(f"Reading covariate file: {covar_file}")
            
        # Read file similar to phenotype
        try:
            # Auto-detect delimiter
            with open(covar_file, 'r') as f:
                first_line = f.readline().strip()
                
            if '\\t' in first_line:
                delimiter = '\\t'
            elif ',' in first_line:
                delimiter = ','
            else:
                delimiter = '\\s+'
                
            if delimiter == '\\s+':
                df = pd.read_csv(covar_file, sep=delimiter, engine='python')
            else:
                df = pd.read_csv(covar_file, sep=delimiter)
                
        except Exception as e:
            raise ValueError(f"Error reading covariate file: {e}")
            
        # Handle sample ID column
        if isinstance(sample_id_col, int):
            if sample_id_col >= len(df.columns):
                raise ValueError(f"Sample ID column index {sample_id_col} out of range")
            sample_id_col = df.columns[sample_id_col]
            
        if sample_id_col not in df.columns:
            raise ValueError(f"Sample ID column '{sample_id_col}' not found")
            
        # Set sample ID as index
        df = df.set_index(sample_id_col)
        
        # Exclude specified columns
        if exclude_cols:
            exclude_cols = [col for col in exclude_cols if col in df.columns]
            df = df.drop(columns=exclude_cols)
            
        # Convert to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
                
        if self.verbose:
            print(f"Loaded {len(df.columns)} covariates for {len(df)} samples")
            
        return df
    
    def create_binary_phenotype(self, phenotypes: np.ndarray,
                               threshold: Optional[float] = None,
                               upper_percentile: float = 90,
                               lower_percentile: float = 10) -> np.ndarray:
        """
        Convert quantitative phenotype to binary (case/control)
        
        Args:
            phenotypes: Quantitative phenotype values
            threshold: Fixed threshold for case/control (if None, use percentiles)
            upper_percentile: Upper percentile for cases
            lower_percentile: Lower percentile for controls
            
        Returns:
            Binary phenotype array (0=control, 1=case, NaN=excluded)
        """
        valid_pheno = phenotypes[~np.isnan(phenotypes)]
        
        if len(valid_pheno) == 0:
            return np.full_like(phenotypes, np.nan)
            
        binary_pheno = np.full_like(phenotypes, np.nan)
        
        if threshold is not None:
            # Use fixed threshold
            binary_pheno[phenotypes >= threshold] = 1
            binary_pheno[phenotypes < threshold] = 0
        else:
            # Use percentiles
            upper_thresh = np.percentile(valid_pheno, upper_percentile)
            lower_thresh = np.percentile(valid_pheno, lower_percentile)
            
            # Cases: upper percentile
            binary_pheno[phenotypes >= upper_thresh] = 1
            
            # Controls: lower percentile  
            binary_pheno[phenotypes <= lower_thresh] = 0
            
            # Exclude middle values
            
        if self.verbose:
            n_cases = np.sum(binary_pheno == 1)
            n_controls = np.sum(binary_pheno == 0)
            n_excluded = np.sum(np.isnan(binary_pheno))
            
            print(f"Created binary phenotype: {n_cases} cases, {n_controls} controls, {n_excluded} excluded")
            
        return binary_pheno
    
    def match_samples(self, phenotype_df: pd.DataFrame,
                     sample_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Match phenotype data to sample list and return aligned arrays
        
        Args:
            phenotype_df: DataFrame with IID and PHENOTYPE columns
            sample_list: List of sample IDs to match
            
        Returns:
            Aligned phenotype array and matched sample list
        """
        # Create dictionary for fast lookup
        pheno_dict = dict(zip(phenotype_df['IID'], phenotype_df['PHENOTYPE']))
        
        # Match samples
        matched_phenotypes = []
        matched_samples = []
        
        for sample_id in sample_list:
            if sample_id in pheno_dict:
                matched_phenotypes.append(pheno_dict[sample_id])
                matched_samples.append(sample_id)
            else:
                matched_phenotypes.append(np.nan)
                matched_samples.append(sample_id)
                
        if self.verbose:
            n_matched = len([p for p in matched_phenotypes if not np.isnan(p)])
            print(f"Matched {n_matched}/{len(sample_list)} samples with phenotype data")
            
        return np.array(matched_phenotypes), matched_samples
    
    def write_phenotype_file(self, sample_ids: List[str],
                           phenotypes: np.ndarray,
                           output_file: str,
                           include_fid: bool = True):
        """
        Write phenotype data to file
        
        Args:
            sample_ids: Sample identifiers
            phenotypes: Phenotype values
            output_file: Output file path
            include_fid: Whether to include FID column (for PLINK compatibility)
        """
        if self.verbose:
            print(f"Writing phenotype file: {output_file}")
            
        # Create DataFrame
        if include_fid:
            # Use same ID for FID and IID (PLINK format)
            pheno_df = pd.DataFrame({
                'FID': sample_ids,
                'IID': sample_ids,
                'PHENOTYPE': phenotypes
            })
        else:
            pheno_df = pd.DataFrame({
                'IID': sample_ids,
                'PHENOTYPE': phenotypes
            })
            
        # Replace NaN with PLINK missing code
        pheno_df['PHENOTYPE'] = pheno_df['PHENOTYPE'].replace(np.nan, -9)
        
        # Write to file
        pheno_df.to_csv(output_file, sep='\\t', index=False)
        
        if self.verbose:
            n_valid = np.sum(~np.isnan(phenotypes))
            print(f"Wrote {n_valid}/{len(phenotypes)} samples with valid phenotypes")
