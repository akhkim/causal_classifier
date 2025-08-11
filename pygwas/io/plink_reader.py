"""
PLINK file reader for PyGWAS
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import struct
import os

class PLINKReader:
    """
    PLINK binary file format reader (.bed, .bim, .fam)
    """
    
    def __init__(self, plink_prefix: str, verbose: bool = True):
        """
        Initialize PLINK reader
        
        Args:
            plink_prefix: Prefix for PLINK files (without extension)
            verbose: Whether to print progress messages
        """
        self.plink_prefix = plink_prefix
        self.verbose = verbose
        
        self.bed_file = f"{plink_prefix}.bed"
        self.bim_file = f"{plink_prefix}.bim" 
        self.fam_file = f"{plink_prefix}.fam"
        
        # Check if files exist
        for file_path in [self.bed_file, self.bim_file, self.fam_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PLINK file not found: {file_path}")
                
    def read_fam(self) -> pd.DataFrame:
        """
        Read .fam file (sample information)
        
        Returns:
            DataFrame with sample information
        """
        fam_columns = ['FID', 'IID', 'FATHER', 'MOTHER', 'SEX', 'PHENOTYPE']
        
        try:
            fam_df = pd.read_csv(self.fam_file, sep='\\s+', header=None, 
                               names=fam_columns, dtype=str)
        except:
            # Fallback for files with different separators
            fam_df = pd.read_csv(self.fam_file, sep=None, header=None,
                               names=fam_columns, dtype=str, engine='python')
            
        return fam_df
    
    def read_bim(self) -> pd.DataFrame:
        """
        Read .bim file (variant information)
        
        Returns:
            DataFrame with variant information
        """
        bim_columns = ['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2']
        
        try:
            bim_df = pd.read_csv(self.bim_file, sep='\\s+', header=None,
                               names=bim_columns, dtype={'CHR': str, 'SNP': str, 
                                                        'CM': float, 'POS': int,
                                                        'A1': str, 'A2': str})
        except:
            # Fallback
            bim_df = pd.read_csv(self.bim_file, sep=None, header=None,
                               names=bim_columns, engine='python')
            
        return bim_df
    
    def read_bed(self, fam_df: pd.DataFrame, bim_df: pd.DataFrame) -> np.ndarray:
        """
        Read .bed file (binary genotype data)
        
        Args:
            fam_df: Sample information from .fam file
            bim_df: Variant information from .bim file
            
        Returns:
            Genotype matrix (samples x variants) with values 0, 1, 2, or NaN
        """
        n_samples = len(fam_df)
        n_variants = len(bim_df)
        
        if self.verbose:
            print(f"Reading BED file: {n_samples} samples, {n_variants} variants")
            
        # Initialize genotype matrix
        genotypes = np.full((n_samples, n_variants), np.nan, dtype=float)
        
        with open(self.bed_file, 'rb') as f:
            # Read magic number (should be 0x6c, 0x1b)
            magic = f.read(2)
            if magic != b'\x6c\x1b':
                raise ValueError("Invalid BED file format")
                
            # Read mode byte
            mode = f.read(1)[0]
            if mode != 1:
                raise ValueError("Only SNP-major mode supported")
                
            # Calculate bytes per variant
            bytes_per_variant = (n_samples + 3) // 4
            
            # Read genotype data
            for variant_idx in range(n_variants):
                if self.verbose and variant_idx % 10000 == 0:
                    print(f"Processing variant {variant_idx}/{n_variants}")
                    
                # Read bytes for this variant
                variant_bytes = f.read(bytes_per_variant)
                
                if len(variant_bytes) != bytes_per_variant:
                    raise ValueError(f"Unexpected end of file at variant {variant_idx}")
                    
                # Decode genotypes
                sample_idx = 0
                for byte in variant_bytes:
                    # Each byte contains 4 genotypes (2 bits each)
                    for shift in [0, 2, 4, 6]:
                        if sample_idx >= n_samples:
                            break
                            
                        # Extract 2-bit genotype
                        geno_code = (byte >> shift) & 3
                        
                        # Convert PLINK coding to dosage
                        # PLINK: 00=hom(A1), 01=missing, 10=het, 11=hom(A2)
                        # Convert to: 0=hom(A1), 1=het, 2=hom(A2), NaN=missing
                        if geno_code == 0:  # Homozygous A1
                            genotypes[sample_idx, variant_idx] = 2
                        elif geno_code == 1:  # Missing
                            genotypes[sample_idx, variant_idx] = np.nan
                        elif geno_code == 2:  # Heterozygous
                            genotypes[sample_idx, variant_idx] = 1
                        elif geno_code == 3:  # Homozygous A2
                            genotypes[sample_idx, variant_idx] = 0
                            
                        sample_idx += 1
                        
        return genotypes
    
    def read_plink(self, max_variants: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Read complete PLINK dataset
        
        Args:
            max_variants: Maximum number of variants to read
            
        Returns:
            Genotype matrix, variant DataFrame, sample DataFrame
        """
        if self.verbose:
            print(f"Reading PLINK files: {self.plink_prefix}")
            
        # Read sample information
        fam_df = self.read_fam()
        
        # Read variant information
        bim_df = self.read_bim()
        
        # Limit variants if requested
        if max_variants:
            bim_df = bim_df.head(max_variants)
            
        # Read genotype data
        genotypes = self.read_bed(fam_df, bim_df)
        
        # Subset genotypes if we limited variants
        if max_variants:
            genotypes = genotypes[:, :max_variants]
            
        if self.verbose:
            print(f"Loaded {genotypes.shape[1]} variants for {genotypes.shape[0]} samples")
            
        return genotypes, bim_df, fam_df
    
    def write_plink(self, genotype_matrix: np.ndarray,
                   variant_df: pd.DataFrame,
                   sample_df: pd.DataFrame,
                   output_prefix: str):
        """
        Write data to PLINK format
        
        Args:
            genotype_matrix: Genotype matrix (samples x variants)
            variant_df: Variant information (must have CHR, SNP, POS, A1, A2)
            sample_df: Sample information (must have FID, IID)
            output_prefix: Output file prefix
        """
        if self.verbose:
            print(f"Writing PLINK files: {output_prefix}")
            
        n_samples, n_variants = genotype_matrix.shape
        
        # Write .fam file
        fam_file = f"{output_prefix}.fam"
        fam_columns = ['FID', 'IID', 'FATHER', 'MOTHER', 'SEX', 'PHENOTYPE']
        
        # Ensure required columns exist
        fam_out = sample_df.copy()
        for col in fam_columns:
            if col not in fam_out.columns:
                if col in ['FATHER', 'MOTHER']:
                    fam_out[col] = '0'
                elif col == 'SEX':
                    fam_out[col] = '0'
                elif col == 'PHENOTYPE':
                    fam_out[col] = '-9'
                    
        fam_out[fam_columns].to_csv(fam_file, sep='\\t', header=False, index=False)
        
        # Write .bim file
        bim_file = f"{output_prefix}.bim"
        bim_columns = ['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2']
        
        bim_out = variant_df.copy()
        if 'CM' not in bim_out.columns:
            bim_out['CM'] = 0
            
        bim_out[bim_columns].to_csv(bim_file, sep='\\t', header=False, index=False)
        
        # Write .bed file
        bed_file = f"{output_prefix}.bed"
        
        with open(bed_file, 'wb') as f:
            # Write magic number and mode
            f.write(b'\\x6c\\x1b\\x01')
            
            # Calculate bytes per variant
            bytes_per_variant = (n_samples + 3) // 4
            
            # Write each variant
            for variant_idx in range(n_variants):
                if self.verbose and variant_idx % 10000 == 0:
                    print(f"Writing variant {variant_idx}/{n_variants}")
                    
                # Pack genotypes into bytes
                variant_bytes = bytearray(bytes_per_variant)
                
                for sample_idx in range(n_samples):
                    byte_idx = sample_idx // 4
                    bit_idx = (sample_idx % 4) * 2
                    
                    dosage = genotype_matrix[sample_idx, variant_idx]
                    
                    # Convert dosage to PLINK coding
                    if np.isnan(dosage):
                        geno_code = 1  # Missing
                    elif dosage == 0:
                        geno_code = 3  # Homozygous A2
                    elif dosage == 1:
                        geno_code = 2  # Heterozygous
                    elif dosage == 2:
                        geno_code = 0  # Homozygous A1
                    else:
                        # Round non-integer dosages
                        if dosage < 0.5:
                            geno_code = 3
                        elif dosage < 1.5:
                            geno_code = 2
                        else:
                            geno_code = 0
                            
                    # Set bits in byte
                    variant_bytes[byte_idx] |= (geno_code << bit_idx)
                    
                f.write(variant_bytes)
                
        if self.verbose:
            print(f"PLINK files written: {output_prefix}.{{bed,bim,fam}}")
    
    def convert_to_text(self, output_prefix: str, 
                       genotype_matrix: np.ndarray,
                       variant_df: pd.DataFrame,
                       sample_df: pd.DataFrame):
        """
        Convert to PLINK text format (.ped and .map)
        
        Args:
            output_prefix: Output file prefix
            genotype_matrix: Genotype matrix
            variant_df: Variant information
            sample_df: Sample information
        """
        if self.verbose:
            print(f"Converting to PLINK text format: {output_prefix}")
            
        # Write .map file
        map_file = f"{output_prefix}.map"
        map_data = variant_df[['CHR', 'SNP', 'CM', 'POS']].copy()
        if 'CM' not in map_data.columns:
            map_data['CM'] = 0
        map_data.to_csv(map_file, sep='\\t', header=False, index=False)
        
        # Write .ped file
        ped_file = f"{output_prefix}.ped"
        
        with open(ped_file, 'w') as f:
            for i, (_, sample) in enumerate(sample_df.iterrows()):
                # Sample information
                ped_fields = [
                    str(sample.get('FID', 'FAM1')),
                    str(sample.get('IID', f'IND{i+1}')),
                    str(sample.get('FATHER', '0')),
                    str(sample.get('MOTHER', '0')),
                    str(sample.get('SEX', '0')),
                    str(sample.get('PHENOTYPE', '-9'))
                ]
                
                # Genotype data
                for j in range(genotype_matrix.shape[1]):
                    dosage = genotype_matrix[i, j]
                    
                    if np.isnan(dosage):
                        ped_fields.extend(['0', '0'])
                    elif dosage == 0:
                        # Homozygous for A2
                        a2 = variant_df.iloc[j]['A2']
                        ped_fields.extend([a2, a2])
                    elif dosage == 1:
                        # Heterozygous
                        a1 = variant_df.iloc[j]['A1']
                        a2 = variant_df.iloc[j]['A2']
                        ped_fields.extend([a1, a2])
                    elif dosage == 2:
                        # Homozygous for A1
                        a1 = variant_df.iloc[j]['A1']
                        ped_fields.extend([a1, a1])
                    else:
                        # Round non-integer dosages
                        if dosage < 0.5:
                            a2 = variant_df.iloc[j]['A2']
                            ped_fields.extend([a2, a2])
                        elif dosage < 1.5:
                            a1 = variant_df.iloc[j]['A1']
                            a2 = variant_df.iloc[j]['A2']
                            ped_fields.extend([a1, a2])
                        else:
                            a1 = variant_df.iloc[j]['A1']
                            ped_fields.extend([a1, a1])
                            
                f.write('\\t'.join(ped_fields) + '\\n')
                
        if self.verbose:
            print(f"PLINK text files written: {output_prefix}.{{ped,map}}")
