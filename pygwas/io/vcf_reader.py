"""
VCF file reader for PyGWAS
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
import gzip
import warnings

class VCFReader:
    """
    VCF (Variant Call Format) file reader
    """
    
    def __init__(self, vcf_file: str, verbose: bool = True):
        """
        Initialize VCF reader
        
        Args:
            vcf_file: Path to VCF file (.vcf or .vcf.gz)
            verbose: Whether to print progress messages
        """
        self.vcf_file = vcf_file
        self.verbose = verbose
        self.samples = []
        self.variants = []
        
    def _open_file(self):
        """Open VCF file (handles both compressed and uncompressed)"""
        if self.vcf_file.endswith('.gz'):
            return gzip.open(self.vcf_file, 'rt')
        else:
            return open(self.vcf_file, 'r')
            
    def _parse_header(self, file_handle) -> List[str]:
        """Parse VCF header and extract sample names"""
        samples = []
        
        for line in file_handle:
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                # Header line with sample names
                fields = line.strip().split('\t')
                if len(fields) > 9:
                    samples = fields[9:]  # Sample names start from column 10
                break
        
        return samples
    
    def _parse_genotype(self, gt_string: str, format_fields: List[str]) -> float:
        """
        Parse genotype string and return dosage (0, 1, 2, or NaN)
        
        Args:
            gt_string: Genotype string from VCF
            format_fields: FORMAT field names
            
        Returns:
            Genotype dosage
        """
        if gt_string == '.':
            return np.nan
            
        # Split by ':'
        gt_values = gt_string.split(':')
        
        # Look for GT (genotype) field
        if 'GT' in format_fields:
            gt_idx = format_fields.index('GT')
            if gt_idx < len(gt_values):
                gt = gt_values[gt_idx]
                
                # Handle different genotype formats
                if '/' in gt:
                    alleles = gt.split('/')
                elif '|' in gt:
                    alleles = gt.split('|')
                else:
                    return np.nan
                    
                # Convert to dosage
                try:
                    dosage = 0
                    for allele in alleles:
                        if allele == '.':
                            return np.nan
                        dosage += int(allele)
                    return float(dosage)
                except ValueError:
                    return np.nan
                    
        # Look for DS (dosage) field
        elif 'DS' in format_fields:
            ds_idx = format_fields.index('DS')
            if ds_idx < len(gt_values):
                try:
                    return float(gt_values[ds_idx])
                except ValueError:
                    return np.nan
                    
        # Look for GP (genotype probabilities) field
        elif 'GP' in format_fields:
            gp_idx = format_fields.index('GP')
            if gp_idx < len(gt_values):
                try:
                    probs = [float(x) for x in gt_values[gp_idx].split(',')]
                    if len(probs) >= 3:
                        # Expected dosage from probabilities
                        dosage = 1 * probs[1] + 2 * probs[2]
                        return dosage
                except ValueError:
                    return np.nan
                    
        return np.nan
    
    def read_vcf(self, max_variants: Optional[int] = None,
                 min_maf: float = 0.0,
                 max_missing: float = 1.0) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        """
        Read VCF file and return genotype matrix
        
        Args:
            max_variants: Maximum number of variants to read
            min_maf: Minimum minor allele frequency filter
            max_missing: Maximum missing rate filter
            
        Returns:
            Genotype matrix (samples x variants), variant info DataFrame, sample names
        """
        if self.verbose:
            print(f"Reading VCF file: {self.vcf_file}")
            
        variants_data = []
        genotypes_list = []
        
        with self._open_file() as f:
            # Parse header
            samples = self._parse_header(f)
            n_samples = len(samples)
            
            if self.verbose:
                print(f"Found {n_samples} samples")
                
            variant_count = 0
            
            # Parse variants
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                    
                # Extract variant information
                chrom = fields[0]
                pos = int(fields[1])
                variant_id = fields[2] if fields[2] != '.' else f"{chrom}:{pos}"
                ref_allele = fields[3]
                alt_allele = fields[4]
                
                # Skip multi-allelic variants for simplicity
                if ',' in alt_allele:
                    continue
                    
                format_fields = fields[8].split(':')
                
                # Parse genotypes for all samples
                genotypes = []
                for i in range(9, len(fields)):
                    if i - 9 < n_samples:
                        gt = self._parse_genotype(fields[i], format_fields)
                        genotypes.append(gt)
                    
                genotypes = np.array(genotypes)
                
                # Apply filters
                valid_geno = ~np.isnan(genotypes)
                missing_rate = 1 - np.mean(valid_geno)
                
                if missing_rate > max_missing:
                    continue
                    
                if np.sum(valid_geno) > 0:
                    maf = np.mean(genotypes[valid_geno]) / 2
                    maf = min(maf, 1 - maf)  # Take minor allele frequency
                    
                    if maf < min_maf:
                        continue
                        
                # Store variant data
                variants_data.append({
                    'CHR': chrom,
                    'POS': pos,
                    'SNP': variant_id,
                    'REF': ref_allele,
                    'ALT': alt_allele,
                    'MAF': maf if 'maf' in locals() else np.nan,
                    'MISSING_RATE': missing_rate
                })
                
                genotypes_list.append(genotypes)
                variant_count += 1
                
                if max_variants and variant_count >= max_variants:
                    break
                    
                if self.verbose and variant_count % 10000 == 0:
                    print(f"Processed {variant_count} variants...")
                    
        if len(genotypes_list) == 0:
            raise ValueError("No variants found after filtering")
            
        # Convert to arrays
        genotype_matrix = np.array(genotypes_list).T  # Transpose to samples x variants
        variant_df = pd.DataFrame(variants_data)
        
        if self.verbose:
            print(f"Loaded {genotype_matrix.shape[1]} variants for {genotype_matrix.shape[0]} samples")
            
        self.samples = samples
        self.variants = variant_df
        
        return genotype_matrix, variant_df, samples
    
    def write_vcf(self, genotype_matrix: np.ndarray,
                  variant_df: pd.DataFrame,
                  sample_names: List[str],
                  output_file: str):
        """
        Write genotype data to VCF format
        
        Args:
            genotype_matrix: Genotype matrix (samples x variants)
            variant_df: Variant information DataFrame
            sample_names: Sample names
            output_file: Output VCF filename
        """
        if self.verbose:
            print(f"Writing VCF file: {output_file}")
            
        # Determine if output should be compressed
        if output_file.endswith('.gz'):
            f = gzip.open(output_file, 'wt')
        else:
            f = open(output_file, 'w')
            
        try:
            # Write VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write("##source=PyGWAS\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            
            # Write column header
            header_fields = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
            header_fields.extend(sample_names)
            f.write('\t'.join(header_fields) + '\n')
            
            # Write variants
            for i, (_, variant) in enumerate(variant_df.iterrows()):
                # Basic variant info
                fields = [
                    str(variant['CHR']),
                    str(variant['POS']),
                    str(variant['SNP']),
                    str(variant['REF']),
                    str(variant['ALT']),
                    '.',  # QUAL
                    'PASS',  # FILTER
                    '.',  # INFO
                    'GT'  # FORMAT
                ]
                
                # Add genotypes
                for j in range(len(sample_names)):
                    dosage = genotype_matrix[j, i]
                    
                    if np.isnan(dosage):
                        gt = './.'
                    elif dosage == 0:
                        gt = '0/0'
                    elif dosage == 1:
                        gt = '0/1'
                    elif dosage == 2:
                        gt = '1/1'
                    else:
                        # Handle non-integer dosages (imputed data)
                        if dosage < 0.5:
                            gt = '0/0'
                        elif dosage < 1.5:
                            gt = '0/1'
                        else:
                            gt = '1/1'
                            
                    fields.append(gt)
                    
                f.write('\t'.join(fields) + '\n')
                
        finally:
            f.close()
            
        if self.verbose:
            print(f"VCF file written with {len(variant_df)} variants")
