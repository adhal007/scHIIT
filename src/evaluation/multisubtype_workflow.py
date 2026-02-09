"""
Multi-Subtype Feature Selection and Evaluation Workflow

This module handles:
1. Running feature selection for multiple subtypes (each vs background)
2. Collecting features from all subtypes
3. Evaluating in common PCA space
4. Comparing methods across subtypes

Author: scHIIT team
Date: 2026-02-09
"""

import numpy as np
import pandas as pd
import time
import gc
from scipy.sparse import csr_matrix, issparse
from typing import Dict, List, Tuple, Optional
from anndata import AnnData
import warnings
warnings.filterwarnings('ignore')


class MultiSubtypeWorkflow:
    """
    Run feature selection for multiple subtypes and evaluate.
    
    Workflow:
    1. For each subtype: Run feature selection (subtype vs background)
    2. Collect all selected features per method
    3. Evaluate on multi-class classification in PCA space
    """
    
    def __init__(
        self,
        adata: AnnData,
        subtypes: List[str],
        tf_list: List[str],
        subtype_column: str = 'Subclass',
        max_background: int = 50000,
        verbose: bool = True
    ):
        """
        Initialize workflow.
        
        Args:
            adata: Full AnnData with all cell types
            subtypes: List of subtype names to analyze
            tf_list: List of TF names to select from
            subtype_column: Column in adata.obs with subtype labels
            max_background: Max background cells per subtype
            verbose: Print progress
        """
        self.adata_full = adata
        self.subtypes = subtypes
        self.tf_list = tf_list
        self.subtype_column = subtype_column
        self.max_background = max_background
        self.verbose = verbose
        
        # Storage for results
        self.feature_results = {}  # {method: {subtype: [features]}}
        self.timing_results = {}   # {method: {subtype: runtime}}
        
        if self.verbose:
            print(f"MultiSubtypeWorkflow initialized:")
            print(f"  Total cells: {adata.n_obs}")
            print(f"  Subtypes to analyze: {len(subtypes)}")
            print(f"  TFs to select from: {len(tf_list)}")
            print(f"  Subtype counts:")
            for st in subtypes:
                n_cells = (adata.obs[subtype_column] == st).sum()
                print(f"    {st}: {n_cells} cells")
    
    def create_subtype_subset(
        self,
        target_subtype: str,
        add_background: bool = True
    ) -> AnnData:
        """
        Create subset with target subtype + background.
        
        Args:
            target_subtype: Target subtype name
            add_background: Whether to add background cells
        
        Returns:
            AnnData subset
        """
        target_mask = self.adata_full.obs[self.subtype_column] == target_subtype
        target_idx = np.where(target_mask)[0]
        
        if add_background:
            # Background = all other subtypes
            background_mask = ~self.adata_full.obs[self.subtype_column].isin(self.subtypes)
            background_idx = np.where(background_mask)[0]
            
            # Subsample background if too large
            if len(background_idx) > self.max_background:
                np.random.seed(42)
                background_idx = np.random.choice(
                    background_idx, self.max_background, replace=False
                )
            
            # Combine
            keep_idx = np.concatenate([target_idx, background_idx])
        else:
            keep_idx = target_idx
        
        # Create subset
        subset = self.adata_full[keep_idx, :].copy()
        
        # Ensure sparse format
        if not issparse(subset.X):
            subset.X = csr_matrix(subset.X)
        
        # Create binary labels for feature selection
        if add_background:
            subset.obs['binary_label'] = [
                target_subtype if i in target_idx else 'background'
                for i in keep_idx
            ]
        
        return subset
    
    def run_feature_selection_all_subtypes(
        self,
        methods: List[str],
        method_params: Optional[Dict] = None,
        gjsd_methods: Optional[List[str]] = None,
        gjsd_params: Optional[Dict] = None
    ):
        """
        Run feature selection for all subtypes using all methods.
        
        Supports both baseline methods and GJSD methods.
        
        Args:
            methods: List of baseline method names (e.g., 'wilcoxon', 'seurat_hvg')
            method_params: Dict of baseline method parameters (applies to ALL baseline methods)
            gjsd_methods: Optional list of GJSD method names (e.g., 'geometric_jsd')
            gjsd_params: Dict of GJSD parameters (applies to ALL GJSD methods)
        """
        method_params = method_params or {}
        gjsd_params = gjsd_params or {}
        gjsd_methods = gjsd_methods or []
        
        # Combine all methods
        all_methods = list(methods) + list(gjsd_methods)
        
        for method_name in all_methods:
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"METHOD: {method_name}")
                print('='*80)
            
            self.feature_results[method_name] = {}
            self.timing_results[method_name] = {}
            
            # Determine if this is a GJSD method or baseline method
            is_gjsd_method = method_name in gjsd_methods
            
            for subtype in self.subtypes:
                if self.verbose:
                    print(f"\n  Subtype: {subtype}")
                
                # Create subset
                adata_subset = self.create_subtype_subset(subtype)
                
                try:
                    start = time.time()
                    
                    if is_gjsd_method:
                        # Run GJSD method
                        selected_features = self._run_gjsd_method(
                            adata_subset, 
                            subtype, 
                            method_name,
                            gjsd_params
                        )
                    else:
                        # Run baseline method
                        selected_features = self._run_baseline_method(
                            adata_subset,
                            subtype,
                            method_name,
                            method_params
                        )
                    
                    runtime = time.time() - start
                    
                    # Store results
                    self.feature_results[method_name][subtype] = selected_features
                    self.timing_results[method_name][subtype] = runtime
                    
                    if self.verbose:
                        print(f"    ✓ Selected {len(selected_features)} features ({runtime:.2f}s)")
                        print(f"      Top 5: {selected_features[:5]}")
                    
                    # Clean up
                    del adata_subset
                    gc.collect()
                    
                except Exception as e:
                    if self.verbose:
                        print(f"    ✗ FAILED: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    self.feature_results[method_name][subtype] = []
                    self.timing_results[method_name][subtype] = 0.0
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("Feature selection complete for all methods and subtypes")
            print('='*80)
    
    def _run_baseline_method(
        self,
        adata_subset: AnnData,
        subtype: str,
        method_name: str,
        method_params: Dict
    ) -> List[str]:
        """
        Run a baseline method (from bm_methods).
        
        Args:
            adata_subset: AnnData subset with target + background
            subtype: Target subtype name
            method_name: Baseline method name
            method_params: Method parameters (can be single dict or dict of dicts)
        
        Returns:
            List of selected features
        """
        from src.methods.bm_methods import get_method
        
        # Check if method_params is a dict of dicts or a single dict
        if method_name in method_params:
            # Per-method parameters
            params = method_params[method_name]
        elif any(isinstance(v, dict) for v in method_params.values()):
            # It's a dict of dicts, but this method not specified
            params = {}
        else:
            # It's a single dict to apply to all methods
            params = method_params
        
        # Create method instance with user params
        method = get_method(
            method_name,
            adata=adata_subset,
            tf_list=self.tf_list,
            target_cell_type=subtype,
            background_cell_type='background',
            cell_type_key='binary_label',
            verbose=False,
            **params  # User params will override defaults
        )
        
        # Run feature selection
        results = method.run()
        selected_features = results['identity_tfs']
        
        # Clean up
        del method
        
        return selected_features
    
    def _run_gjsd_method(
        self,
        adata_subset: AnnData,
        subtype: str,
        jsd_method: str,
        gjsd_params: Dict
    ) -> List[str]:
        """
        Run a GJSD method (from schiit_method).
        
        Args:
            adata_subset: AnnData subset with target + background
            subtype: Target subtype name
            jsd_method: GJSD method name (e.g., 'bidirectional_gjsd')
            gjsd_params: GJSD parameters
        
        Returns:
            List of selected features
        """
        import src.methods.schiit_method.tf_filters.jsd_filter as schiit_main
        
        # Default GJSD parameters
        default_params = {
            'chipseq_file': None,
            'scgx_sig_file': None,
            'main_filter': 'unique_only',
            'expr_method': None,
            'top_n_high': None,
            'top_jsd_pc': None,
            'top_n_jsd': 1000,
            'identity_top_percent': 10.0,
            'verbose': False
        }
        
        # Update with user-provided parameters
        default_params.update(gjsd_params)
        
        # Create pipeline
        pipeline = schiit_main.CoreFilterOnlyPipeline(
            adata=adata_subset,
            tf_list=self.tf_list,
            target_cell_type=subtype,
            background_cell_type='background',
            cell_type_key='binary_label',
            jsd_method=jsd_method,
            **default_params
        )
        
        # Run
        pipeline.run()
        selected_features = pipeline.results['identity_tfs']
        
        # Clean up
        del pipeline
        
        return selected_features
    
    def get_union_features(self, method_name: str) -> List[str]:
        """
        Get union of all features selected across subtypes for a method.
        
        Args:
            method_name: Name of the method
        
        Returns:
            List of unique features
        """
        if method_name not in self.feature_results:
            return []
        
        all_features = set()
        for subtype_features in self.feature_results[method_name].values():
            all_features.update(subtype_features)
        
        return list(all_features)
    
    def evaluate_on_multiclass(
        self,
        n_pcs: int = 10,
        n_neighbors: int = 15,
        use_pca: bool = True,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Evaluate all methods on multi-class classification.
        
        Uses union of features from all subtypes, projects to PCA space,
        then evaluates classification of all subtypes.
        
        Args:
            n_pcs: Number of principal components
            n_neighbors: Number of neighbors for kNN
            use_pca: Whether to use PCA
            random_state: Random seed
        
        Returns:
            DataFrame with evaluation metrics
        """
        from evaluation_metrics import FeatureEvaluator
        
        # Create dataset with only the subtypes we analyzed
        adata_eval = self.adata_full[
            self.adata_full.obs[self.subtype_column].isin(self.subtypes)
        ].copy()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("MULTI-CLASS EVALUATION")
            print('='*80)
            print(f"Dataset: {adata_eval.n_obs} cells, {len(self.subtypes)} subtypes")
            print(f"PCA: {n_pcs} components" if use_pca else "No PCA")
        
        # Initialize evaluator
        evaluator = FeatureEvaluator(
            adata_eval,
            cell_type_key=self.subtype_column,
            verbose=False
        )
        
        results = []
        
        for method_name in self.feature_results.keys():
            # Get union of features
            union_features = self.get_union_features(method_name)
            
            if len(union_features) == 0:
                if self.verbose:
                    print(f"\n  ✗ {method_name}: No features selected")
                continue
            
            if self.verbose:
                print(f"\n  {method_name}: {len(union_features)} features (union)")
            
            # Evaluate
            metrics = evaluator.evaluate_features(
                selected_features=union_features,
                method_name=method_name,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                use_pca=use_pca,
                random_state=random_state
            )
            
            # Add feature union info
            metrics['n_features_union'] = len(union_features)
            
            # Add timing info
            total_time = sum(self.timing_results[method_name].values())
            metrics['total_runtime'] = total_time
            metrics['avg_runtime_per_subtype'] = total_time / len(self.subtypes)
            
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # Sort by accuracy
        if 'knn_accuracy' in df.columns:
            df = df.sort_values('knn_accuracy', ascending=False)
        
        return df
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics of feature selection across subtypes.
        
        Returns:
            DataFrame with statistics per method
        """
        summary = []
        
        for method_name in self.feature_results.keys():
            # Features per subtype
            n_features_per_subtype = [
                len(features) 
                for features in self.feature_results[method_name].values()
            ]
            
            # Union features
            union_features = self.get_union_features(method_name)
            
            # Timing
            runtimes = list(self.timing_results[method_name].values())
            
            summary.append({
                'method': method_name,
                'n_features_mean': np.mean(n_features_per_subtype),
                'n_features_std': np.std(n_features_per_subtype),
                'n_features_min': np.min(n_features_per_subtype),
                'n_features_max': np.max(n_features_per_subtype),
                'n_features_union': len(union_features),
                'runtime_total': np.sum(runtimes),
                'runtime_mean_per_subtype': np.mean(runtimes),
            })
        
        return pd.DataFrame(summary)
    
    def compute_feature_overlap(self, top_k: int = 50) -> pd.DataFrame:
        """
        Compute pairwise feature overlap between methods.
        
        Args:
            top_k: Use top K features per subtype
        
        Returns:
            DataFrame with Jaccard similarities
        """
        from itertools import combinations
        
        overlaps = []
        
        for m1, m2 in combinations(self.feature_results.keys(), 2):
            # Get top K features for each subtype
            features_m1 = set()
            features_m2 = set()
            
            for subtype in self.subtypes:
                if subtype in self.feature_results[m1]:
                    features_m1.update(
                        self.feature_results[m1][subtype][:top_k]
                    )
                if subtype in self.feature_results[m2]:
                    features_m2.update(
                        self.feature_results[m2][subtype][:top_k]
                    )
            
            # Jaccard similarity
            intersection = len(features_m1 & features_m2)
            union = len(features_m1 | features_m2)
            jaccard = intersection / union if union > 0 else 0
            
            overlaps.append({
                'method_1': m1,
                'method_2': m2,
                'intersection': intersection,
                'union': union,
                'jaccard': jaccard
            })
        
        return pd.DataFrame(overlaps)
    
    def save_results(self, output_dir: str = '.'):
        """
        Save all results to CSV files.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Feature lists per method per subtype
        for method_name, subtype_features in self.feature_results.items():
            df_features = pd.DataFrame([
                {'subtype': subtype, 'rank': i+1, 'feature': feat}
                for subtype, features in subtype_features.items()
                for i, feat in enumerate(features)
            ])
            
            filepath = os.path.join(output_dir, f'features_{method_name}.csv')
            df_features.to_csv(filepath, index=False)
        
        # 2. Summary statistics
        df_summary = self.get_summary_statistics()
        df_summary.to_csv(
            os.path.join(output_dir, 'summary_statistics.csv'),
            index=False
        )
        
        # 3. Feature overlap
        df_overlap = self.compute_feature_overlap()
        df_overlap.to_csv(
            os.path.join(output_dir, 'feature_overlap.csv'),
            index=False
        )
        
        if self.verbose:
            print(f"\n✓ Results saved to {output_dir}/")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_multisubtype_benchmark(
    adata: AnnData,
    subtypes: List[str],
    tf_list: List[str],
    methods: List[str],
    gjsd_methods: Optional[List[str]] = None,
    method_params: Optional[Dict] = None,
    subtype_column: str = 'Subclass',
    max_background: int = 50000,
    n_pcs: int = 10,
    gjsd_params: Optional[Dict] = None,
    output_dir: str = './multisubtype_results',
    verbose: bool = True
) -> Tuple[pd.DataFrame, MultiSubtypeWorkflow]:
    """
    Complete multi-subtype benchmark in one function.
    
    Supports both baseline methods and GJSD methods.
    
    Args:
        adata: AnnData with all cell types
        subtypes: List of subtypes to analyze
        tf_list: List of TFs
        methods: List of baseline method names (e.g., ['wilcoxon', 'seurat_hvg'])
        gjsd_methods: Optional list of GJSD method names (e.g., ['geometric_jsd'])
        method_params: Optional dict of baseline method parameters
        subtype_column: Column with subtype labels
        max_background: Max background cells
        n_pcs: Number of PCs for evaluation
        gjsd_params: Optional dict of GJSD-specific parameters
        output_dir: Where to save results
        verbose: Print progress
    
    Returns:
        Tuple of (evaluation_df, workflow_object)
    """
    # Initialize workflow
    workflow = MultiSubtypeWorkflow(
        adata=adata,
        subtypes=subtypes,
        tf_list=tf_list,
        subtype_column=subtype_column,
        max_background=max_background,
        verbose=verbose
    )
    
    # Run feature selection (both baseline and GJSD)
    workflow.run_feature_selection_all_subtypes(
        methods=methods,
        method_params=method_params,
        gjsd_methods=gjsd_methods,
        gjsd_params=gjsd_params
    )
    
    # Evaluate
    eval_df = workflow.evaluate_on_multiclass(
        n_pcs=n_pcs,
        use_pca=True,
        random_state=42
    )
    
    # Print summary
    if verbose:
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print('='*80)
        
        # Show available columns dynamically
        display_cols = ['method']
        optional_cols = ['knn_accuracy', 'knn_f1_macro', 'ari', 'nmi', 
                        'n_features_union', 'n_features', 'total_runtime']
        
        for col in optional_cols:
            if col in eval_df.columns:
                display_cols.append(col)
        
        print(eval_df[display_cols].to_string(index=False))
    
    # Save results
    workflow.save_results(output_dir)
    eval_df.to_csv(f'{output_dir}/evaluation_results.csv', index=False)
    
    return eval_df, workflow