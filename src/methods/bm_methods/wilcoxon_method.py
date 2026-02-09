"""
Wilcoxon differential expression using Scanpy's implementation.

Uses scanpy.tl.rank_genes_groups with method='wilcoxon'.
This is the standard Scanpy/Seurat differential expression method.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict
from .base_method import BaseBenchmarkMethod


class WilcoxonMethod(BaseBenchmarkMethod):
    """
    Wilcoxon rank-sum test using Scanpy implementation.
    
    Uses scanpy.tl.rank_genes_groups(method='wilcoxon')
    This is the standard implementation used in Scanpy and Seurat.
    
    Reference:
        Widely used in scRNA-seq DE analysis
        Implemented in scanpy.tl.rank_genes_groups
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Use Scanpy's Wilcoxon implementation.
        
        Runs differential expression: target vs background.
        Uses -log10(pval_adj) * log2(fold_change) as score.
        
        Returns:
            Dictionary of TF -> combined score
        """
        # Create temporary AnnData with target and background
        adata_temp = self.adata.copy()
        
        # Subset to TFs only for speed
        adata_temp = adata_temp[:, self.tf_names].copy()
        
        # Ensure cell type labels are set
        # (should already be in 'Class' column)
        
        # Run Scanpy's Wilcoxon test
        try:
            sc.tl.rank_genes_groups(
                adata_temp,
                groupby=self.cell_type_key,
                groups=[self.target_cell_type],
                reference=self.background_cell_type,
                method='wilcoxon',
                use_raw=False,
                key_added='rank_genes_wilcoxon'
            )
            
            # Extract results for target cell type
            result = sc.get.rank_genes_groups_df(
                adata_temp, 
                group=self.target_cell_type,
                key='rank_genes_wilcoxon'
            )
            
            # Compute combined score: -log10(pval_adj) * |log2FC|
            scores = {}
            for _, row in result.iterrows():
                gene_name = row['names']
                pval_adj = row['pvals_adj']
                logfc = row['logfoldchanges']
                
                # Avoid log(0)
                if pval_adj == 0:
                    pval_adj = 1e-300
                elif pval_adj >= 1:
                    pval_adj = 1 - 1e-10
                
                # Combined score
                score = -np.log10(pval_adj) * abs(logfc)
                scores[gene_name] = score
            
            # Ensure all TFs have scores
            for tf_name in self.tf_names:
                if tf_name not in scores:
                    scores[tf_name] = 0.0
                    
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Scanpy Wilcoxon failed, using scipy: {e}")
            # Fallback to scipy implementation
            from scipy.stats import ranksums
            scores = {}
            for tf_idx, tf_name in zip(self.tf_indices, self.tf_names):
                expr_target = self.X_target[:, tf_idx]
                expr_background = self.X_background[:, tf_idx]
                
                mean_target = np.mean(expr_target) + 1e-9
                mean_background = np.mean(expr_background) + 1e-9
                log2fc = np.log2(mean_target / mean_background)
                
                try:
                    stat, pval = ranksums(expr_target, expr_background)
                    if pval == 0:
                        pval = 1e-300
                    score = -np.log10(pval) * abs(log2fc)
                except:
                    score = 0.0
                
                scores[tf_name] = score
        
        return scores
