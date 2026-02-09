"""
Mutual Information feature selection.

Measures dependency between gene expression and cell type label.
Information-theoretic approach for feature selection.
"""

import numpy as np
from typing import Dict
from sklearn.feature_selection import mutual_info_classif
from .base_method import BaseBenchmarkMethod


class MutualInfoMethod(BaseBenchmarkMethod):
    """
    Mutual information between gene expression and cell type.
    
    MI measures how much knowing a gene's expression tells you
    about the cell type, and vice versa.
    
    MI(X; Y) = H(X) + H(Y) - H(X,Y)
    
    where H is entropy.
    
    Reference:
        Cover & Thomas (2006) Elements of Information Theory
        sklearn.feature_selection.mutual_info_classif
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute mutual information between each TF and cell type label.
        
        Creates binary labels (target vs background) and computes
        MI between expression and labels.
        
        Returns:
            Dictionary of TF -> MI score
        """
        scores = {}
        
        # Create binary labels: 1 for target, 0 for background
        n_target = self.X_target.shape[0]
        n_background = self.X_background.shape[0]
        labels = np.concatenate([
            np.ones(n_target),
            np.zeros(n_background)
        ])
        
        # Combined expression matrix
        X_combined = np.vstack([self.X_target, self.X_background])
        
        # Extract TF expression
        X_tfs = X_combined[:, self.tf_indices]
        
        # Compute MI for all TFs at once (more efficient)
        try:
            mi_scores = mutual_info_classif(
                X_tfs, 
                labels, 
                discrete_features=False,
                random_state=42,
                n_neighbors=3
            )
            
            # Map to gene names
            for tf_name, mi_score in zip(self.tf_names, mi_scores):
                scores[tf_name] = mi_score
                
        except Exception as e:
            if self.verbose:
                print(f"  Warning: MI computation failed: {e}")
            # Fallback: compute per-gene
            for tf_idx, tf_name in zip(self.tf_indices, self.tf_names):
                try:
                    mi = mutual_info_classif(
                        X_combined[:, [tf_idx]], 
                        labels, 
                        discrete_features=False,
                        random_state=42
                    )[0]
                    scores[tf_name] = mi
                except:
                    scores[tf_name] = 0.0
        
        return scores
