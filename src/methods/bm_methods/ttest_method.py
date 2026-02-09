"""
T-test for differential expression.

Parametric statistical test for identifying genes with different
mean expression between target and background.
"""

import numpy as np
from typing import Dict
from scipy.stats import ttest_ind
from .base_method import BaseBenchmarkMethod


class TTestMethod(BaseBenchmarkMethod):
    """
    Differential expression using t-test.
    
    Classic parametric test for comparing means between two groups.
    Assumes normality but works well in practice for scRNA-seq.
    
    Reference:
        Student's t-test (1908)
        Commonly used in bulk and single-cell DE analysis
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute t-statistic combined with fold-change.
        
        Score = |t-statistic| * sign(log2FC)
        
        Returns:
            Dictionary of TF -> t-score
        """
        scores = {}
        
        for tf_idx, tf_name in zip(self.tf_indices, self.tf_names):
            expr_target = self.X_target[:, tf_idx]
            expr_background = self.X_background[:, tf_idx]
            
            # Compute fold change for direction
            mean_target = np.mean(expr_target) + 1e-9
            mean_background = np.mean(expr_background) + 1e-9
            log2fc = np.log2(mean_target / mean_background)
            
            # T-test (independent samples)
            try:
                stat, pval = ttest_ind(expr_target, expr_background, equal_var=False)
                
                # Score: absolute t-statistic (effect size)
                # Multiply by sign of fold change for direction
                score = abs(stat) * np.sign(log2fc)
                
                # If t-test fails (e.g., zero variance), score = fold change
                if np.isnan(score) or np.isinf(score):
                    score = abs(log2fc)
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: T-test failed for {tf_name}: {e}")
                score = abs(log2fc)
            
            scores[tf_name] = abs(score)  # Use absolute for ranking
        
        return scores
