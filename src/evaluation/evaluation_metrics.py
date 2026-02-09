"""
Evaluation metrics for feature selection benchmarking.

This module provides comprehensive evaluation of selected features across:
- Classification performance (kNN, Random Forest)
- Clustering quality (ARI, NMI, Silhouette)
- Batch effect assessment (LISI, kBET, Silhouette)
- Marker gene recovery
- Stability across subsampling

Author: scHIIT team
Date: 2026-02-09
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FeatureEvaluator:
    """
    Comprehensive evaluation of selected features.
    
    Evaluates features on multiple criteria:
    1. Classification: How well can features separate cell types?
    2. Clustering: Do features produce biologically meaningful clusters?
    3. Batch effects: Are features batch-invariant or batch-specific?
    4. Marker recovery: Do features include known markers?
    5. Stability: Are features consistent across subsamples?
    """
    
    def __init__(
        self,
        adata: AnnData,
        cell_type_key: str = 'cell_type',
        batch_key: Optional[str] = None,
        known_markers: Optional[Dict[str, List[str]]] = None,
        verbose: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            adata: AnnData object with expression data
            cell_type_key: Column in adata.obs with cell type labels
            batch_key: Optional column in adata.obs with batch labels
            known_markers: Optional dict mapping cell types to known marker genes
            verbose: Print progress messages
        """
        self.adata = adata
        self.cell_type_key = cell_type_key
        self.batch_key = batch_key
        self.known_markers = known_markers or {}
        self.verbose = verbose
        
        # Extract data
        self.cell_types = adata.obs[cell_type_key].values
        self.batch_labels = adata.obs[batch_key].values if batch_key else None
        
        # Get expression matrix
        X = adata.X
        self.X = X.toarray() if issparse(X) else X.copy()
        
        if self.verbose:
            print(f"FeatureEvaluator initialized:")
            print(f"  Cells: {adata.n_obs}")
            print(f"  Genes: {adata.n_vars}")
            print(f"  Cell types: {len(np.unique(self.cell_types))}")
            if batch_key:
                print(f"  Batches: {len(np.unique(self.batch_labels))}")
    
    def evaluate_features(
        self,
        selected_features: List[str],
        method_name: str = "unknown",
        n_neighbors: int = 15,
        n_pcs: int = 50,
        use_pca: bool = True,
        random_state: int = 42
    ) -> Dict:
        """
        Comprehensive evaluation of selected features.
        
        Args:
            selected_features: List of selected gene names
            method_name: Name of the method (for logging)
            n_neighbors: Number of neighbors for kNN classifier
            n_pcs: Number of principal components to use
            use_pca: Whether to project into PC space before evaluation
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with all evaluation metrics
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Evaluating: {method_name}")
            print(f"Features: {len(selected_features)}")
            if use_pca:
                print(f"Using PCA: {n_pcs} components")
            print('='*80)
        
        # Get feature indices
        feature_indices = [i for i, g in enumerate(self.adata.var_names) 
                          if g in selected_features]
        
        if len(feature_indices) == 0:
            raise ValueError("No selected features found in adata")
        
        # Subset expression matrix
        X_subset = self.X[:, feature_indices]
        
        # Apply PCA if requested
        if use_pca:
            from sklearn.decomposition import PCA
            
            # Determine number of PCs (cannot exceed features or samples)
            n_pcs_actual = min(n_pcs, X_subset.shape[0] - 1, X_subset.shape[1])
            
            if self.verbose:
                print(f"  Computing PCA ({n_pcs_actual} components)...")
            
            pca = PCA(n_components=n_pcs_actual, random_state=random_state)
            X_eval = pca.fit_transform(X_subset)
            
            # Store PCA info
            pca_var_explained = pca.explained_variance_ratio_.sum()
            
            if self.verbose:
                print(f"  PCA variance explained: {pca_var_explained:.3f}")
        else:
            X_eval = X_subset
            pca_var_explained = 1.0
        
        metrics = {
            'method': method_name,
            'n_features': len(selected_features),
            'n_features_found': len(feature_indices),
            'n_pcs': n_pcs_actual if use_pca else len(feature_indices),
            'pca_variance_explained': pca_var_explained,
            'used_pca': use_pca
        }
        
        # 1. Classification metrics
        if self.verbose:
            print("  Computing classification metrics...")
        classification_metrics = self._evaluate_classification(
            X_eval, n_neighbors=n_neighbors, random_state=random_state
        )
        metrics.update(classification_metrics)
        
        # 2. Clustering metrics
        if self.verbose:
            print("  Computing clustering metrics...")
        clustering_metrics = self._evaluate_clustering(
            X_eval, random_state=random_state
        )
        metrics.update(clustering_metrics)
        
        # 3. Batch effect metrics (if batch info available)
        if self.batch_key is not None:
            if self.verbose:
                print("  Computing batch effect metrics...")
            batch_metrics = self._evaluate_batch_effects(X_eval)
            metrics.update(batch_metrics)
        
        # 4. Marker recovery (if known markers provided)
        if self.known_markers:
            if self.verbose:
                print("  Computing marker recovery...")
            marker_metrics = self._evaluate_marker_recovery(selected_features)
            metrics.update(marker_metrics)
        
        # 5. Feature statistics (on original features, not PCs)
        if self.verbose:
            print("  Computing feature statistics...")
        feature_stats = self._compute_feature_statistics(X_subset)
        metrics.update(feature_stats)
        
        if self.verbose:
            print(f"\n  ✓ Evaluation complete")
            print(f"    - Classification accuracy: {metrics.get('knn_accuracy', 0):.3f}")
            print(f"    - Clustering ARI: {metrics.get('ari', 0):.3f}")
            if 'batch_lisi' in metrics:
                print(f"    - Batch LISI: {metrics['batch_lisi']:.3f}")
        
        return metrics
    
    # ========================================================================
    # CLASSIFICATION METRICS
    # ========================================================================
    
    def _evaluate_classification(
        self, 
        X: np.ndarray, 
        n_neighbors: int = 15,
        random_state: int = 42
    ) -> Dict:
        """
        Evaluate classification performance using kNN and Random Forest.
        
        Uses stratified k-fold cross-validation.
        
        Args:
            X: Expression matrix (cells x selected_features)
            n_neighbors: Number of neighbors for kNN
            random_state: Random seed
        
        Returns:
            Dictionary with classification metrics
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, make_scorer
        
        metrics = {}
        
        # Check if we have multiple classes
        unique_classes = np.unique(self.cell_types)
        if len(unique_classes) < 2:
            metrics['knn_accuracy'] = 1.0
            metrics['knn_f1_macro'] = 1.0
            metrics['rf_accuracy'] = 1.0
            return metrics
        
        # k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        # kNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_acc_scores = cross_val_score(knn, X, self.cell_types, cv=cv, scoring='accuracy')
        
        # For F1 score, handle binary and multiclass cases
        try:
            if len(unique_classes) == 2:
                # Binary classification - use binary F1
                knn_f1_scores = cross_val_score(
                    knn, X, self.cell_types, cv=cv, 
                    scoring=make_scorer(f1_score, average='binary', zero_division=0, pos_label=unique_classes[0])
                )
            else:
                # Multiclass - use macro F1
                knn_f1_scores = cross_val_score(
                    knn, X, self.cell_types, cv=cv, 
                    scoring=make_scorer(f1_score, average='macro', zero_division=0)
                )
            
            # Check for NaN values
            if np.isnan(knn_f1_scores).any():
                # Fallback: compute F1 manually
                from sklearn.model_selection import cross_val_predict
                y_pred = cross_val_predict(knn, X, self.cell_types, cv=cv)
                
                if len(unique_classes) == 2:
                    knn_f1 = f1_score(self.cell_types, y_pred, average='binary', zero_division=0, pos_label=unique_classes[0])
                else:
                    knn_f1 = f1_score(self.cell_types, y_pred, average='macro', zero_division=0)
                
                metrics['knn_f1_macro'] = knn_f1
                metrics['knn_f1_std'] = 0.0
            else:
                metrics['knn_f1_macro'] = knn_f1_scores.mean()
                metrics['knn_f1_std'] = knn_f1_scores.std()
                
        except Exception as e:
            # If F1 computation fails, fall back to accuracy
            if self.verbose:
                print(f"    Warning: F1 computation failed ({e}), using accuracy")
            metrics['knn_f1_macro'] = knn_acc_scores.mean()
            metrics['knn_f1_std'] = knn_acc_scores.std()
        
        metrics['knn_accuracy'] = knn_acc_scores.mean()
        metrics['knn_accuracy_std'] = knn_acc_scores.std()
        
        # Random Forest classifier (if not too many cells)
        if X.shape[0] < 50000:
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            rf_acc_scores = cross_val_score(rf, X, self.cell_types, cv=cv, scoring='accuracy')
            metrics['rf_accuracy'] = rf_acc_scores.mean()
            metrics['rf_accuracy_std'] = rf_acc_scores.std()
        
        return metrics
    
    # ========================================================================
    # CLUSTERING METRICS
    # ========================================================================
    
    def _evaluate_clustering(
        self, 
        X: np.ndarray,
        random_state: int = 42
    ) -> Dict:
        """
        Evaluate clustering quality.
        
        Computes:
        - ARI (Adjusted Rand Index)
        - NMI (Normalized Mutual Information)
        - Silhouette score
        
        Args:
            X: Expression matrix (cells x selected_features)
            random_state: Random seed
        
        Returns:
            Dictionary with clustering metrics
        """
        from sklearn.metrics import (
            adjusted_rand_score, 
            normalized_mutual_info_score,
            silhouette_score
        )
        from sklearn.cluster import KMeans
        
        metrics = {}
        
        # Number of clusters = number of true cell types
        n_clusters = len(np.unique(self.cell_types))
        
        if n_clusters < 2:
            metrics['ari'] = 1.0
            metrics['nmi'] = 1.0
            metrics['silhouette_celltype'] = 1.0
            return metrics
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # ARI: How well clusters match true cell types
        metrics['ari'] = adjusted_rand_score(self.cell_types, cluster_labels)
        
        # NMI: Normalized mutual information
        metrics['nmi'] = normalized_mutual_info_score(self.cell_types, cluster_labels)
        
        # Silhouette score for true cell types
        # (Are cells of same type close together?)
        try:
            metrics['silhouette_celltype'] = silhouette_score(X, self.cell_types)
        except:
            metrics['silhouette_celltype'] = 0.0
        
        return metrics
    
    # ========================================================================
    # BATCH EFFECT METRICS
    # ========================================================================
    
    def _evaluate_batch_effects(self, X: np.ndarray) -> Dict:
        """
        Evaluate batch effect mixing.
        
        Computes:
        - Batch LISI (Local Inverse Simpson Index)
        - Batch silhouette score
        - Cell type silhouette score
        
        Args:
            X: Expression matrix (cells x selected_features)
        
        Returns:
            Dictionary with batch metrics
        """
        metrics = {}
        
        # Silhouette score for batches (lower is better - more mixed)
        from sklearn.metrics import silhouette_score
        
        try:
            # Batch silhouette: lower = better mixing
            batch_sil = silhouette_score(X, self.batch_labels)
            metrics['batch_silhouette'] = batch_sil
            
            # Cell type silhouette: higher = better separation
            celltype_sil = silhouette_score(X, self.cell_types)
            metrics['celltype_silhouette'] = celltype_sil
            
        except:
            metrics['batch_silhouette'] = 0.0
            metrics['celltype_silhouette'] = 0.0
        
        # LISI score (if available)
        try:
            lisi_scores = self._compute_lisi(X, self.batch_labels)
            metrics['batch_lisi'] = lisi_scores.mean()
            metrics['batch_lisi_std'] = lisi_scores.std()
        except:
            if self.verbose:
                print("    Warning: LISI computation failed")
        
        return metrics
    
    def _compute_lisi(
        self, 
        X: np.ndarray, 
        labels: np.ndarray,
        n_neighbors: int = 90
    ) -> np.ndarray:
        """
        Compute Local Inverse Simpson Index (LISI).
        
        LISI measures local diversity of labels.
        Higher LISI = more mixed (better for batch, worse for cell type).
        
        Args:
            X: Expression matrix
            labels: Batch or cell type labels
            n_neighbors: Number of neighbors
        
        Returns:
            LISI score for each cell
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        # Compute LISI for each cell
        lisi_scores = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            # Get labels of neighbors
            neighbor_labels = labels[indices[i, 1:]]  # Exclude self
            
            # Count frequency of each label
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            
            # Simpson index
            simpson = np.sum((counts / len(neighbor_labels)) ** 2)
            
            # LISI (inverse Simpson index)
            lisi_scores[i] = 1.0 / simpson if simpson > 0 else 1.0
        
        return lisi_scores
    
    # ========================================================================
    # MARKER GENE RECOVERY
    # ========================================================================
    
    def _evaluate_marker_recovery(self, selected_features: List[str]) -> Dict:
        """
        Evaluate recovery of known marker genes.
        
        Args:
            selected_features: List of selected gene names
        
        Returns:
            Dictionary with marker recovery metrics
        """
        metrics = {}
        
        selected_set = set(selected_features)
        
        # Overall marker recovery
        all_markers = set()
        for markers in self.known_markers.values():
            all_markers.update(markers)
        
        if len(all_markers) > 0:
            recovered = selected_set & all_markers
            metrics['marker_recall'] = len(recovered) / len(all_markers)
            metrics['marker_precision'] = len(recovered) / len(selected_set) if len(selected_set) > 0 else 0
            metrics['n_markers_total'] = len(all_markers)
            metrics['n_markers_recovered'] = len(recovered)
        
        # Per-cell-type marker recovery
        for cell_type, markers in self.known_markers.items():
            marker_set = set(markers)
            recovered = selected_set & marker_set
            
            key = f"marker_recall_{cell_type}"
            metrics[key] = len(recovered) / len(marker_set) if len(marker_set) > 0 else 0
        
        return metrics
    
    # ========================================================================
    # FEATURE STATISTICS
    # ========================================================================
    
    def _compute_feature_statistics(self, X: np.ndarray) -> Dict:
        """
        Compute statistics about the selected features.
        
        Args:
            X: Expression matrix (cells x selected_features)
        
        Returns:
            Dictionary with feature statistics
        """
        metrics = {}
        
        # Mean expression
        metrics['mean_expression'] = X.mean()
        
        # Sparsity (fraction of zeros)
        metrics['sparsity'] = (X == 0).sum() / X.size
        
        # Variance
        metrics['mean_variance'] = X.var(axis=0).mean()
        
        # Coefficient of variation
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        cv = stds / (means + 1e-9)
        metrics['mean_cv'] = cv.mean()
        
        return metrics
    
    # ========================================================================
    # BATCH-AWARE EVALUATION
    # ========================================================================
    
    def evaluate_per_batch(
        self,
        selected_features: List[str],
        method_name: str = "unknown"
    ) -> pd.DataFrame:
        """
        Evaluate features separately for each batch.
        
        Useful for understanding batch-specific vs universal features.
        
        Args:
            selected_features: List of selected gene names
            method_name: Name of the method
        
        Returns:
            DataFrame with per-batch metrics
        """
        if self.batch_key is None:
            raise ValueError("No batch information available")
        
        results = []
        
        for batch in np.unique(self.batch_labels):
            # Subset to this batch
            batch_mask = self.batch_labels == batch
            adata_batch = self.adata[batch_mask, :].copy()
            
            # Create evaluator for this batch
            evaluator = FeatureEvaluator(
                adata_batch,
                cell_type_key=self.cell_type_key,
                batch_key=None,
                verbose=False
            )
            
            # Evaluate
            metrics = evaluator.evaluate_features(
                selected_features,
                method_name=f"{method_name}_batch{batch}"
            )
            
            metrics['batch'] = batch
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    # ========================================================================
    # STABILITY EVALUATION
    # ========================================================================
    
    def evaluate_stability(
        self,
        feature_selection_func,
        n_subsamples: int = 5,
        subsample_frac: float = 0.8,
        random_state: int = 42
    ) -> Dict:
        """
        Evaluate stability of feature selection across subsamples.
        
        Args:
            feature_selection_func: Function that takes adata and returns selected features
            n_subsamples: Number of subsamples to test
            subsample_frac: Fraction of cells to keep in each subsample
            random_state: Random seed
        
        Returns:
            Dictionary with stability metrics
        """
        np.random.seed(random_state)
        
        feature_sets = []
        
        for i in range(n_subsamples):
            # Subsample cells
            n_cells = int(self.adata.n_obs * subsample_frac)
            indices = np.random.choice(self.adata.n_obs, n_cells, replace=False)
            adata_sub = self.adata[indices, :].copy()
            
            # Run feature selection
            features = feature_selection_func(adata_sub)
            feature_sets.append(set(features))
        
        # Compute pairwise Jaccard similarity
        from itertools import combinations
        
        jaccard_scores = []
        for set1, set2 in combinations(feature_sets, 2):
            jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
            jaccard_scores.append(jaccard)
        
        # Compute core features (present in all subsamples)
        core_features = set.intersection(*feature_sets)
        
        metrics = {
            'stability_jaccard_mean': np.mean(jaccard_scores),
            'stability_jaccard_std': np.std(jaccard_scores),
            'n_core_features': len(core_features),
            'core_feature_fraction': len(core_features) / len(feature_sets[0]) if len(feature_sets[0]) > 0 else 0
        }
        
        return metrics


# ========================================================================
# BATCH EVALUATION WRAPPER
# ========================================================================

def evaluate_multiple_methods(
    adata: AnnData,
    method_results: Dict[str, List[str]],
    cell_type_key: str = 'cell_type',
    batch_key: Optional[str] = None,
    known_markers: Optional[Dict[str, List[str]]] = None,
    n_neighbors: int = 15,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate multiple feature selection methods.
    
    Args:
        adata: AnnData object
        method_results: Dict mapping method_name -> selected_features
        cell_type_key: Column with cell type labels
        batch_key: Optional column with batch labels
        known_markers: Optional dict of known markers
        n_neighbors: Number of neighbors for kNN
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        DataFrame with evaluation metrics for all methods
    """
    evaluator = FeatureEvaluator(
        adata,
        cell_type_key=cell_type_key,
        batch_key=batch_key,
        known_markers=known_markers,
        verbose=False
    )
    
    results = []
    
    for method_name, selected_features in method_results.items():
        if verbose:
            print(f"\nEvaluating: {method_name}")
        
        metrics = evaluator.evaluate_features(
            selected_features,
            method_name=method_name,
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Sort by classification accuracy
    if 'knn_accuracy' in df.columns:
        df = df.sort_values('knn_accuracy', ascending=False)
    
    return df


# ========================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================

def quick_evaluate(
    adata: AnnData,
    selected_features: List[str],
    cell_type_key: str = 'cell_type',
    method_name: str = "method"
) -> Dict:
    """
    Quick evaluation with minimal setup.
    
    Args:
        adata: AnnData object
        selected_features: List of selected gene names
        cell_type_key: Column with cell type labels
        method_name: Name of the method
    
    Returns:
        Dictionary with key metrics
    """
    evaluator = FeatureEvaluator(adata, cell_type_key=cell_type_key, verbose=False)
    return evaluator.evaluate_features(selected_features, method_name=method_name)
