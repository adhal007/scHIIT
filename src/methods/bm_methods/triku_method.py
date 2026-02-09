"""
Triku feature selection method.

Triku is a recent method specifically designed for identifying
highly variable genes in scRNA-seq data.

Reference:
    Lebrigand et al. (2023) NAR Genomics and Bioinformatics
    https://doi.org/10.1093/nargab/lqad064
    
Installation:
    pip install triku
"""

import numpy as np
import pandas as pd
from typing import Dict
from .base_method import BaseBenchmarkMethod
import warnings


class TrikuMethod(BaseBenchmarkMethod):
    """
    Triku feature selection.
    
    Uses the Triku package for robust HVG detection.
    Falls back to error if package not installed.
    
    Reference:
        Lebrigand et al. (2023) NAR Genomics and Bioinformatics
        
    Implementation:
        pip install triku
        https://github.com/alexmascension/triku
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute Triku scores for feature selection.
        
        Returns:
            Dictionary of TF -> Triku score
        """
        try:
            import triku as tk
        except ImportError:
            raise ImportError(
                "Triku package not installed. "
                "Install with: pip install triku"
            )
        
        # Create temporary AnnData with only target cells
        adata_target = self.adata[self.target_mask, :].copy()
        
        # Subset to TFs only
        adata_tfs = adata_target[:, self.tf_names].copy()
        
        try:
            # Run Triku
            # Triku adds scores to adata.var['triku_score']
            tk.tl.triku(adata_tfs, use_raw=False)
            
            # Extract scores
            scores = {}
            for gene_name in self.tf_names:
                if gene_name in adata_tfs.var_names and 'triku_scores' in adata_tfs.var.columns:
                    score = adata_tfs.var.loc[gene_name, 'triku_scores']
                    scores[gene_name] = score
                else:
                    scores[gene_name] = 0.0
            
            if self.verbose:
                print("  Using Triku package")
                
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Triku failed: {e}")
            # Return zero scores
            scores = {tf_name: 0.0 for tf_name in self.tf_names}
        
        return scores
