"""
Simple fold-change feature selection.

Most basic differential expression metric.
Compares mean expression between target and background.
"""

import numpy as np
from typing import Dict
from .base_method import BaseBenchmarkMethod


class FoldChangeMethod(BaseBenchmarkMethod):
    """
    Simple fold-change between target and background.
    
    Most basic differential expression metric.
    Often used as a baseline comparison.
    
    FC = Mean(target) / Mean(background)
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute log2 fold-change for each TF.
        
        log2FC = log2(Mean_target / Mean_background)
        
        Returns:
            Dictionary of TF -> |log2FC|
        """
        scores = {}
        
        for tf_idx, tf_name in zip(self.tf_indices, self.tf_names):
            expr_target = self.X_target[:, tf_idx]
            expr_background = self.X_background[:, tf_idx]
            
            # Compute means (add pseudocount to avoid division by zero)
            mean_target = np.mean(expr_target) + 1e-9
            mean_background = np.mean(expr_background) + 1e-9
            
            # Log2 fold change
            log2fc = np.log2(mean_target / mean_background)
            
            # Use absolute fold change for ranking
            scores[tf_name] = abs(log2fc)
        
        return scores
