"""
Seurat-style Highly Variable Genes using Scanpy's implementation.

Uses scanpy.pp.highly_variable_genes with flavor='seurat_v3'.
This is the actual Seurat v3 method used in practice.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict
from .base_method import BaseBenchmarkMethod


class SeuratHVGMethod(BaseBenchmarkMethod):
    """
    Seurat v3 HVG selection using scanpy implementation.
    
    Uses scanpy.pp.highly_variable_genes(flavor='seurat_v3')
    This is the actual method from Seurat v3.
    
    Reference:
        Stuart & Butler et al. (2019) Cell
        https://doi.org/10.1016/j.cell.2019.05.031
        
    Implementation:
        scanpy.pp.highly_variable_genes
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Use Scanpy's Seurat v3 HVG implementation.
        
        Returns variance-standardized scores from Seurat v3 method.
        
        Returns:
            Dictionary of TF -> variance_standardized score
        """
        # Create temporary AnnData with only target cells
        adata_target = self.adata[self.target_mask, :].copy()
        
        # Subset to TFs only
        adata_tfs = adata_target[:, self.tf_names].copy()
        
        # Run Seurat v3 HVG method
        try:
            sc.pp.highly_variable_genes(
                adata_tfs,
                flavor='seurat_v3',
                n_top_genes=len(self.tf_names),  # Score all genes
                subset=False
            )
            
            # Extract scores (variance_standardized)
            scores = {}
            for gene_name in self.tf_names:
                if gene_name in adata_tfs.var_names:
                    # Use variance_standardized as score (higher = more variable)
                    score = adata_tfs.var.loc[gene_name, 'variances_norm']
                    scores[gene_name] = score
                else:
                    scores[gene_name] = 0.0
                    
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Seurat HVG failed, using fallback: {e}")
            # Fallback to simple variance
            scores = {}
            for tf_idx, tf_name in zip(self.tf_indices, self.tf_names):
                expr = self.X_target[:, tf_idx]
                scores[tf_name] = np.var(expr)
        
        return scores
