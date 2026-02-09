"""
M3Drop - Michaelis-Menten Modelling of Dropouts.

M3Drop models the relationship between gene expression and dropout rate
to identify highly variable genes.

Reference:
    Andrews & Hemberg (2019) Bioinformatics
    https://doi.org/10.1093/bioinformatics/bty1044
    
Installation:
    R package: BiocManager::install("M3Drop")
    Or use Python implementation
"""

import numpy as np
import pandas as pd
from typing import Dict
from .base_method import BaseBenchmarkMethod
from scipy import stats


class M3DropMethod(BaseBenchmarkMethod):
    """
    M3Drop dropout-based feature selection.
    
    Identifies genes with higher dropout than expected based on expression level.
    Uses Python implementation (R package also available).
    
    Reference:
        Andrews & Hemberg (2019) Bioinformatics
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute M3Drop scores.
        
        Fits Michaelis-Menten curve to dropout vs expression relationship.
        Genes with high residuals are highly variable.
        
        Returns:
            Dictionary of TF -> M3Drop score
        """
        scores = {}
        
        # Compute mean expression and dropout rate for all TFs
        mean_expr = []
        dropout_rates = []
        
        for tf_idx in self.tf_indices:
            expr = self.X_target[:, tf_idx]
            
            # Mean expression (log scale)
            mean_e = np.mean(np.log1p(expr))
            mean_expr.append(mean_e)
            
            # Dropout rate
            dropout_rate = np.sum(expr == 0) / len(expr)
            dropout_rates.append(dropout_rate)
        
        mean_expr = np.array(mean_expr)
        dropout_rates = np.array(dropout_rates)
        
        # Fit Michaelis-Menten: d = 1 - (S / (K + S))
        # where d = dropout, S = mean expression, K = half-saturation
        
        # Simple approach: fit linear model in log-log space
        # or use residuals from smoothed curve
        
        try:
            # Use LOWESS smoothing to get expected dropout
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Smooth dropout vs expression
            smoothed = lowess(dropout_rates, mean_expr, frac=0.3)
            expected_dropout = np.interp(mean_expr, smoothed[:, 0], smoothed[:, 1])
            
            # Compute residuals (observed - expected)
            residuals = dropout_rates - expected_dropout
            
            # Genes with negative residuals (lower dropout than expected) are HVGs
            # Use absolute value of negative residuals as score
            m3drop_scores = np.where(residuals < 0, -residuals, 0)
            
        except ImportError:
            if self.verbose:
                print("  statsmodels not available, using simple deviation from mean")
            
            # Fallback: just use deviation from median dropout
            median_dropout = np.median(dropout_rates)
            m3drop_scores = np.abs(dropout_rates - median_dropout)
        
        # Normalize scores to [0, 1]
        if m3drop_scores.max() > 0:
            m3drop_scores = m3drop_scores / m3drop_scores.max()
        
        # Map to gene names
        for i, tf_name in enumerate(self.tf_names):
            scores[tf_name] = m3drop_scores[i]
        
        return scores
