"""
Q-Q plot visualization for GWAS results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class QQPlot:
    """Q-Q plot for GWAS p-values"""
    
    def __init__(self, association_results):
        self.results = association_results
    
    def plot(self, output_file=None, title="Q-Q Plot"):
        """
        Create a Q-Q plot
        
        Args:
            output_file: Path to save the plot
            title: Plot title
        """
        print(f"Creating Q-Q plot: {title}")
        
        if self.results is None or len(self.results) == 0:
            print("No association results available for plotting")
            return None
        
        # Calculate genomic inflation factor
        p_values = self.results['P'].dropna()
        if len(p_values) > 0:
            chi2_obs = -2 * np.log(p_values)
            chi2_exp = np.percentile(chi2_obs, np.arange(1, 101))
            lambda_gc = np.median(chi2_obs) / np.log(2)
            
            print(f"- Genomic inflation factor (λ): {lambda_gc:.3f}")
            print(f"- Total p-values: {len(p_values)}")
            print(f"- Min p-value: {p_values.min():.2e}")
        
        return None
