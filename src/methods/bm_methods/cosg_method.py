"""
COSG (Cosine Similarity-based marker Gene identification).

COSG is a recent method for identifying marker genes in scRNA-seq data
using cosine similarity.

Reference:
    Dai et al. (2022) Briefings in Bioinformatics
    https://doi.org/10.1093/bib/bbab579
    
Installation:
    pip install COSG
"""

import numpy as np
import pandas as pd
from typing import Dict
from .base_method import BaseBenchmarkMethod


class COSGMethod(BaseBenchmarkMethod):
    """
    COSG marker gene identification.
    
    Uses cosine similarity to identify cell-type specific markers.
    
    Reference:
        Dai et al. (2022) Briefings in Bioinformatics
        
    Implementation:
        pip install COSG
        https://github.com/genecell/COSG
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute COSG scores for marker identification.
        
        Returns:
            Dictionary of TF -> COSG score
        """
        try:
            import COSG
        except ImportError:
            raise ImportError(
                "COSG package not installed. "
                "Install with: pip install COSG"
            )
        
        # Create temporary AnnData
        adata_temp = self.adata.copy()
        
        # Subset to TFs only for speed
        adata_temp = adata_temp[:, self.tf_names].copy()
        
        try:
            # Run COSG
            # COSG identifies markers for each group
            marker_dict = COSG.cosg(
                adata_temp,
                key_added='cosg_scores',
                mu=1,
                expressed_pct=0.1,
                n_genes_user=len(self.tf_names),  # Return all genes
                remove=False
            )
            
            # Extract scores for target cell type
            scores = {}
            
            if self.target_cell_type in marker_dict['names']:
                target_markers = marker_dict['names'][self.target_cell_type]
                target_scores = marker_dict['scores'][self.target_cell_type]
                
                # Map scores to gene names
                for gene_name, score in zip(target_markers, target_scores):
                    if gene_name in self.tf_names:
                        scores[gene_name] = score
            
            # Ensure all TFs have scores
            for tf_name in self.tf_names:
                if tf_name not in scores:
                    scores[tf_name] = 0.0
            
            if self.verbose:
                print("  Using COSG package")
                
        except Exception as e:
            if self.verbose:
                print(f"  Warning: COSG failed: {e}")
            # Return zero scores
            scores = {tf_name: 0.0 for tf_name in self.tf_names}
        
        return scores
