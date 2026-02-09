"""
Scanpy Highly Variable Genes using native implementation.

Uses scanpy.pp.highly_variable_genes with flavor='seurat'.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict
from .base_method import BaseBenchmarkMethod


class ScanpyHVGMethod(BaseBenchmarkMethod):
    """
    Scanpy HVG selection using native implementation.
    
    Uses scanpy.pp.highly_variable_genes(flavor='seurat')
    This is the standard Scanpy HVG method.
    
    Reference:
        Wolf et al. (2018) Genome Biology
        https://doi.org/10.1186/s13059-017-1382-0
        
    Implementation:
        scanpy.pp.highly_variable_genes
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Use Scanpy's native HVG implementation.
        
        Returns dispersions_norm (normalized dispersion scores).
        
        Returns:
            Dictionary of TF -> dispersion_norm score
        """
        # Create temporary AnnData with only target cells
        adata_target = self.adata[self.target_mask, :].copy()
        
        # Subset to TFs only
        adata_tfs = adata_target[:, self.tf_names].copy()
        
        # Run Scanpy HVG method
        try:
            sc.pp.highly_variable_genes(
                adata_tfs,
                flavor='seurat',
                n_top_genes=len(self.tf_names),  # Score all genes
                subset=False
            )
            
            # Extract scores (dispersions_norm)
            scores = {}
            for gene_name in self.tf_names:
                if gene_name in adata_tfs.var_names:
                    # Use dispersions_norm as score
                    score = adata_tfs.var.loc[gene_name, 'dispersions_norm']
                    scores[gene_name] = score
                else:
                    scores[gene_name] = 0.0
                    
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Scanpy HVG failed, using fallback: {e}")
            # Fallback to variance of log-normalized
            scores = {}
            for tf_idx, tf_name in zip(self.tf_indices, self.tf_names):
                expr = self.X_target[:, tf_idx]
                log_expr = np.log1p(expr)
                scores[tf_name] = np.var(log_expr)
        
        return scores
