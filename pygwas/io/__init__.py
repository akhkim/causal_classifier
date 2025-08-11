"""
Input/Output modules for PyGWAS
"""

from .vcf_reader import VCFReader
from .plink_reader import PLINKReader  
from .phenotype_reader import PhenotypeReader

__all__ = [
    'VCFReader',
    'PLINKReader', 
    'PhenotypeReader'
]
