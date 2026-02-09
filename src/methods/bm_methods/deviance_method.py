"""
Deviance-based feature selection using scry package.

Uses the scry R package for computing deviance residuals.
Falls back to Python implementation if scry is not available.
"""

import numpy as np
import pandas as pd
from typing import Dict
from .base_method import BaseBenchmarkMethod
import warnings


class DevianceMethod(BaseBenchmarkMethod):
    """
    Deviance-based feature selection using scry.
    
    Attempts to use the R scry package via rpy2.
    Falls back to Python Poisson deviance if scry unavailable.
    
    Reference:
        Townes et al. (2019) Genome Biology
        https://doi.org/10.1186/s13059-019-1861-6
        
    Implementation:
        scry R package: https://bioconductor.org/packages/scry/
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute deviance using scry package or fallback.
        
        Returns:
            Dictionary of TF -> deviance score
        """
        # Try using scry R package
        try:
            scores = self._compute_with_scry()
            if self.verbose:
                print("  Using scry R package")
            return scores
        except ImportError:
            if self.verbose:
                print("  scry R package not available, using Python fallback")
            return self._compute_with_python()
        except Exception as e:
            if self.verbose:
                print(f"  scry failed ({e}), using Python fallback")
            return self._compute_with_python()
    
    def _compute_with_scry(self) -> Dict[str, float]:
        """
        Compute deviance using scry R package.
        
        Requires:
            - rpy2
            - R
            - scry Bioconductor package
        
        Returns:
            Dictionary of TF -> deviance score
        """
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        
        # Activate automatic conversion
        pandas2ri.activate()
        numpy2ri.activate()
        
        # Import R packages
        base = importr('base')
        scry = importr('scry')
        singlecellexperiment = importr('SingleCellExperiment')
        
        # Create temporary AnnData with only target cells and TFs
        adata_target = self.adata[self.target_mask, self.tf_names].copy()
        
        # Get count matrix (scry expects counts, not normalized)
        # Assuming adata.X is log-normalized, we need raw counts
        if 'counts' in adata_target.layers:
            counts = adata_target.layers['counts']
        else:
            # If no raw counts, use current X (may not be ideal)
            warnings.warn("No raw counts found, using current X matrix")
            counts = adata_target.X
        
        # Convert to dense if sparse
        from scipy.sparse import issparse
        if issparse(counts):
            counts = counts.toarray()
        
        # Convert to R matrix
        r_counts = ro.r.matrix(counts.T, nrow=counts.shape[1], ncol=counts.shape[0])
        
        # Create SingleCellExperiment
        ro.r.assign('counts_matrix', r_counts)
        ro.r('sce <- SingleCellExperiment(assays = list(counts = counts_matrix))')
        
        # Compute deviance
        ro.r('sce <- scry::devianceFeatureSelection(sce, assay="counts")')
        
        # Extract deviance scores
        ro.r('deviance_scores <- rowData(sce)$binomial_deviance')
        deviance_array = np.array(ro.r('deviance_scores'))
        
        # Map to gene names
        scores = {}
        for i, gene_name in enumerate(self.tf_names):
            scores[gene_name] = deviance_array[i]
        
        return scores
    
    def _compute_with_python(self) -> Dict[str, float]:
        """
        Fallback: Compute Poisson deviance in Python.
        
        This is a simplified version without the full scry model.
        
        Returns:
            Dictionary of TF -> deviance score
        """
        scores = {}
        
        # Compute overall mean expression per gene (null model)
        overall_mean = np.mean(self.X_target, axis=0)
        
        for tf_idx, tf_name in zip(self.tf_indices, self.tf_names):
            expr = self.X_target[:, tf_idx]
            
            # Expected counts under null (Poisson with mean)
            mu = overall_mean[tf_idx]
            
            if mu <= 1e-10:
                scores[tf_name] = 0.0
                continue
            
            # Compute Poisson deviance
            # D = 2 * sum(observed * log(observed/expected) - (observed - expected))
            observed = expr
            expected = np.full_like(expr, mu, dtype=float)
            
            # Handle zeros (avoid log(0))
            mask = observed > 0
            deviance = np.zeros_like(observed, dtype=float)
            
            if mask.any():
                deviance[mask] = 2 * (
                    observed[mask] * np.log(observed[mask] / expected[mask]) - 
                    (observed[mask] - expected[mask])
                )
            
            # For zeros, deviance = 2 * mu
            deviance[~mask] = 2 * mu
            
            # Sum of absolute deviance
            total_deviance = np.sum(np.abs(deviance))
            
            scores[tf_name] = total_deviance
        
        return scores
