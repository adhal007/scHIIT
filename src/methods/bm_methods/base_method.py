"""
Base class for benchmark feature selection methods.

All benchmark methods inherit from this base class to ensure
consistent interface with the GJSD pipeline for fair comparison.

This is separate from src.methods.schiit_method.tf_filters.base_filter
to keep benchmarking code isolated.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from typing import Dict, List, Optional
import time
import warnings

warnings.filterwarnings('ignore')


class BaseBenchmarkMethod:
    """
    Base class for benchmark feature selection methods.
    
    Matches the interface of CoreFilterOnlyPipeline to enable fair comparison.
    All benchmark methods should inherit from this class.
    """
    
    def __init__(
        self,
        adata: AnnData,
        tf_list: List[str],
        target_cell_type: str,
        background_cell_type: str,
        cell_type_key: str = 'cell_type',
        verbose: bool = True,
        top_n_high: int = None,
        top_n_jsd: int = 1000,
        identity_top_percent: float = 10.0,
        identity_top_n: int = None,
    ):
        """
        Initialize benchmark method.
        
        Args:
            adata: AnnData object with expression data
            tf_list: List of TF/gene names to select from
            target_cell_type: Target cell type for feature selection
            background_cell_type: Background/reference cell type
            cell_type_key: Column in adata.obs with cell type labels
            verbose: Print progress messages
            top_n_high: Number of top expressed features (for high expression filter)
            top_n_jsd: Number of features after uniqueness filter (matches GJSD pipeline)
            identity_top_percent: Percentage of top features for identity selection
            identity_top_n: Absolute number of identity features (overrides percent)
        """
        # Core data
        self.adata = adata
        self.tf_list = tf_list
        self.target_cell_type = target_cell_type
        self.background_cell_type = background_cell_type
        self.cell_type_key = cell_type_key
        self.verbose = verbose
        
        # Selection parameters (match GJSD pipeline)
        self.top_n_high = top_n_high
        self.top_n_jsd = top_n_jsd
        self.identity_top_percent = identity_top_percent
        self.identity_top_n = identity_top_n
        
        # Extract cell masks
        self.target_mask = adata.obs[cell_type_key] == target_cell_type
        self.background_mask = adata.obs[cell_type_key] == background_cell_type
        
        # Get expression matrices
        X = adata.X
        self.X = X.toarray() if issparse(X) else X.copy()
        self.X_target = self.X[self.target_mask, :]
        self.X_background = self.X[self.background_mask, :]
        
        # Get gene indices and names for TFs
        self.tf_indices = [i for i, g in enumerate(adata.var_names) if g in tf_list]
        self.tf_names = [adata.var_names[i] for i in self.tf_indices]
        
        # Results storage (matches GJSD pipeline structure)
        self.results = {
            'high_exp_tfs': None,
            'unique_exp_tfs': None,
            'core_filtered_tfs': None,
            'final_filtered_tfs': None,
            'network': None,
            'identity_tfs': None,
            'feature_scores': {},  # Store raw scores
            'jsd_scores': {},      # Renamed scores for compatibility
            'metrics': {}
        }
        
        if self.verbose:
            self._print_initialization()
    
    def run(self):
        """
        Run the feature selection pipeline.
        
        This method orchestrates the full pipeline:
        1. Compute feature scores
        2. Rank features
        3. Apply filters
        4. Select identity TFs
        
        Returns:
            self.results dictionary
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RUNNING: {self.__class__.__name__}")
            print('='*80)
        
        # Step 1: Compute scores for all TFs
        scores = self.compute_scores()
        self.results['feature_scores'] = scores
        
        # Step 2: Rank features by scores
        ranked_tfs = self.rank_features(scores)
        
        # Step 3: Apply high expression filter (if specified)
        if self.top_n_high is not None:
            high_exp_tfs = ranked_tfs[:self.top_n_high]
            self.results['high_exp_tfs'] = high_exp_tfs
            if self.verbose:
                print(f"  High expression filter: {len(high_exp_tfs)} TFs")
        else:
            high_exp_tfs = ranked_tfs
            self.results['high_exp_tfs'] = high_exp_tfs
        
        # Step 4: Apply uniqueness/specificity filter
        # For baselines, this is the same as high expression
        # (they don't have separate uniqueness filter like GJSD)
        self.results['unique_exp_tfs'] = high_exp_tfs
        
        # Step 5: Core filtered TFs (intersection for GJSD, same for baselines)
        n_core = min(self.top_n_jsd, len(high_exp_tfs)) if self.top_n_jsd else len(high_exp_tfs)
        core_filtered_tfs = high_exp_tfs[:n_core]
        self.results['core_filtered_tfs'] = core_filtered_tfs
        
        if self.verbose:
            print(f"  Core filtered TFs: {len(core_filtered_tfs)}")
        
        # Step 6: Final filtered (no additional filtering for baselines)
        self.results['final_filtered_tfs'] = core_filtered_tfs
        
        # Step 7: Select identity TFs (top percentage)
        if self.identity_top_n is not None:
            n_identity = min(self.identity_top_n, len(core_filtered_tfs))
        elif self.identity_top_percent is not None:
            n_identity = max(1, int(len(core_filtered_tfs) * self.identity_top_percent / 100))
        else:
            n_identity = max(20, len(core_filtered_tfs) // 5)  # Default: top 20%
        
        identity_tfs = core_filtered_tfs[:n_identity]
        self.results['identity_tfs'] = identity_tfs
        
        if self.verbose:
            print(f"  Identity TFs selected: {len(identity_tfs)}")
        
        # Step 8: Store scores in GJSD-compatible format
        # Convert to dict with gene names as keys for compatibility
        self.results['jsd_scores'] = {gene: scores[gene] for gene in self.tf_names if gene in scores}
        
        # Compute metrics
        runtime = time.time() - start_time
        self.results['metrics'] = self._compute_metrics(runtime)
        
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute feature scores.
        
        MUST BE IMPLEMENTED BY SUBCLASSES.
        
        Returns:
            Dictionary mapping gene names to scores.
            Higher scores = more important features.
        """
        raise NotImplementedError("Subclasses must implement compute_scores()")
    
    def rank_features(self, scores: Dict[str, float]) -> List[str]:
        """
        Rank features by scores.
        
        Args:
            scores: Dictionary of gene -> score
        
        Returns:
            List of gene names sorted by score (descending)
        """
        sorted_genes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [gene for gene, score in sorted_genes]
    
    def _compute_metrics(self, runtime: float) -> Dict:
        """Compute metrics matching GJSD pipeline."""
        metrics = {
            'runtime': runtime,
            'n_input_tfs': len(self.tf_list),
            'n_high_exp': len(self.results['high_exp_tfs'] or []),
            'n_unique_exp': len(self.results['unique_exp_tfs'] or []),
            'n_core_filtered': len(self.results['core_filtered_tfs'] or []),
            'n_final_filtered': len(self.results['final_filtered_tfs'] or []),
            'n_identity_tfs': len(self.results['identity_tfs'] or [])
        }
        return metrics
    
    def _print_initialization(self):
        """Print initialization info."""
        print("="*80)
        print(f"INITIALIZED: {self.__class__.__name__}")
        print("="*80)
        print(f"Target cell type: {self.target_cell_type}")
        print(f"Background cell type: {self.background_cell_type}")
        print(f"Target cells: {self.target_mask.sum()}")
        print(f"Background cells: {self.background_mask.sum()}")
        print(f"Total cells: {self.adata.n_obs}")
        print(f"Input TFs: {len(self.tf_list)}")
    
    def _print_summary(self):
        """Print pipeline results summary."""
        m = self.results['metrics']
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"  Input TFs:        {m['n_input_tfs']:4d}")
        print(f"  Core filtered:    {m['n_core_filtered']:4d}")
        print(f"  Identity TFs:     {m['n_identity_tfs']:4d}")
        print(f"  Runtime:          {m['runtime']:.2f}s")
