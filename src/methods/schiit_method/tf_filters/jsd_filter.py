
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import warnings
from typing import Dict, List, Tuple, Optional
import src.methods.schiit_method.tf_filters.base_filter as tf_base
from src.methods.schiit_method.pareto_gjsd.pareto_slt import ParetoGeneSelector 
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
        Select top TFs by expression uniqueness.
        
        For bidirectional_gjsd/asymmetric_gjsd: Use NSGA-II multi-objective optimization
        For other methods: Use simple JSD score ranking
        """
        if 'jsd_scores' not in self.results or not self.results['jsd_scores']:
            # Fallback: just return all filtered TFs (or top N if too many)
            selected_tfs = filtered_tfs[:min(20, len(filtered_tfs))]
            if self.verbose:
                print(f"  No JSD scores available, returning {len(selected_tfs)} TFs")
            return selected_tfs
        
        # ========================================================================
        # BIDIRECTIONAL/ASYMMETRIC GJSD: Use NSGA-II Multi-Objective
        # ========================================================================
        
        if self.jsd_method in ('bidirectional_gjsd', 'asymmetric_gjsd'):
            if self.verbose:
                print(f"  Using NSGA-II multi-objective optimization for {self.jsd_method}")
            
            try:
                # Get directional scores as DataFrame
                scores_df = self.get_directional_scores_df()
                
                # Filter to only the filtered_tfs
                scores_df = scores_df[scores_df['gene'].isin(filtered_tfs)]
                
                if len(scores_df) == 0:
                    if self.verbose:
                        print(f"  No scores available for filtered TFs")
                    return filtered_tfs[:min(20, len(filtered_tfs))]
                
                # Determine how many to select
                if hasattr(self, 'identity_top_percent') and self.identity_top_percent is not None:
                    n_select = max(1, int(len(filtered_tfs) * self.identity_top_percent / 100))
                elif hasattr(self, 'identity_top_n') and self.identity_top_n is not None:
                    n_select = min(self.identity_top_n, len(filtered_tfs))
                else:
                    n_select = max(20, len(filtered_tfs) // 5)
                
                # Initialize Pareto selector
                 # Import if in separate file
                # OR if in same file, just instantiate directly
                
                selector = ParetoGeneSelector(
                    min_direction=0.0,      # Allow both up and down (bidirectional)
                    min_mean_target=0.1     # No minimum (already filtered)
                )
                
                # Use NSGA-II with crowding distance
                selected_tfs = selector.select_top_genes(
                    scores_df=scores_df,
                    top_n=n_select,
                    ranking_method='crowding_distance'
                )
                
                if self.verbose:
                    print(f"  NSGA-II selected {len(selected_tfs)} TFs")
                
                return selected_tfs
                
            except Exception as e:
                if self.verbose:
                    print(f"  NSGA-II failed: {e}")
                    print(f"  Falling back to simple ranking by signed_specificity")
                
                # Fallback: rank by signed_specificity
                scores = self.results['jsd_scores']
                tf_scores = [(tf, scores[tf][2]) for tf in filtered_tfs if tf in scores]  # index 2 = signed_specificity
                ranked = sorted(tf_scores, key=lambda x: abs(x[1]), reverse=True)
                
                if hasattr(self, 'identity_top_percent') and self.identity_top_percent is not None:
                    n_select = max(1, int(len(ranked) * self.identity_top_percent / 100))
                elif hasattr(self, 'identity_top_n') and self.identity_top_n is not None:
                    n_select = min(self.identity_top_n, len(ranked))
                else:
                    n_select = max(20, len(ranked) // 5)
                
                selected_tfs = [tf for tf, _ in ranked[:n_select]]
                return selected_tfs
        
        # ========================================================================
        # OTHER JSD METHODS: Simple Score Ranking
        # ========================================================================
        else:
            scores = self.results['jsd_scores']
            
            # Get scores for filtered TFs
            tf_scores = [(tf, scores.get(tf, 0)) for tf in filtered_tfs]
            
            # Sort by score (ascending for JSD - lower divergence is more similar)
            ranked = sorted(tf_scores, key=lambda x: x[1], reverse=False)
            
            # Determine how many to select
            if hasattr(self, 'identity_top_percent') and self.identity_top_percent is not None:
                n_select = max(1, int(len(ranked) * self.identity_top_percent / 100))
            elif hasattr(self, 'identity_top_n') and self.identity_top_n is not None:
                n_select = min(self.identity_top_n, len(ranked))
            else:
                n_select = max(20, len(ranked) // 5)

            selected_tfs = [tf for tf, _ in ranked[:n_select]]

            if self.verbose:
                print(f"  Identity TF selection: {len(selected_tfs)} TFs")
                if ranked and n_select > 0:
                    print(f"  Score range: {ranked[0][1]:.3f} - {ranked[n_select-1][1]:.3f}")
            
            return selected_tfs