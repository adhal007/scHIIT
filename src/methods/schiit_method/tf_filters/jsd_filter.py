
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import warnings
from typing import Dict, List, Tuple, Optional
import src.methods.schiit_method.tf_filters.base_filter as tf_base
warnings.filterwarnings('ignore')
class CoreFilterOnlyPipeline(tf_base.BaseTFIdentityPipeline):
    """
    Minimal pipeline using ONLY core filters (high expression + uniqueness).
    NO network analysis - this serves as the baseline for comparison.
    """
    
    def apply_additional_filters(self, core_tfs: List[str]) -> List[str]:
        """No additional filtering - return core TFs as-is."""
        if self.verbose:
            print(f"  No additional filtering applied")
        return core_tfs
    
    def build_network(self, filtered_tfs: List[str]) -> nx.DiGraph:
        """Return empty network - no network analysis for baseline."""
        G = nx.DiGraph()
        # Just add nodes, no edges
        G.add_nodes_from(filtered_tfs)
        
        if self.verbose:
            print(f"  No network built (baseline uses core filters only)")
        
        return G
    
    def identify_key_tfs(self, graph: nx.DiGraph, filtered_tfs: List[str]) -> List[str]:
        """
        Select top TFs by expression uniqueness (JSD scores).
        No network-based selection.
        """
        # Use JSD scores to rank TFs
        if 'jsd_scores' in self.results and self.results['jsd_scores']:
            scores = self.results['jsd_scores']
            
            # Get scores for filtered TFs
            tf_scores = [(tf, scores.get(tf, 0)) for tf in filtered_tfs]
            
            # Sort by score
            ranked = sorted(tf_scores, key=lambda x: x[1], reverse=False)
            
            # Use configurable selection method
            if hasattr(self, 'identity_top_percent') and self.identity_top_percent is not None:
                # Use percentage
                n_select = max(1, int(len(ranked) * self.identity_top_percent / 100))
            elif hasattr(self, 'identity_top_n') and self.identity_top_n is not None:
                # Use absolute number
                n_select = min(self.identity_top_n, len(ranked))
            else:
                # Default: top 20 or 20% (whichever is larger) - preserves old behavior
                n_select = max(20, len(ranked) // 5)

            selected_tfs = [tf for tf, _ in ranked[:n_select]]

            if self.verbose:
                print(f"  Identity TF selection: {len(selected_tfs)} TFs")
                        
            # if self.verbose:
            #     print(f"  Selected top {len(selected_tfs)} TFs by JSD score")
            #     print(f"  Score range: {ranked[0][1]:.3f} - {ranked[n_select-1][1]:.3f}")
        else:
            # Fallback: just return all filtered TFs (or top N if too many)
            selected_tfs = filtered_tfs[:min(20, len(filtered_tfs))]
            
            if self.verbose:
                print(f"  No JSD scores available, returning {len(selected_tfs)} TFs")
        
        return selected_tfs