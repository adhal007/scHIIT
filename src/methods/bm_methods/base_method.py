"""
Base class for benchmarking cell identity methods.

This provides a unified interface for all marker gene selection methods
to ensure consistent evaluation and comparison.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from anndata import AnnData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseBenchmarkMethod(ABC):
    """
    Abstract base class for cell identity / marker gene selection methods.
    
    All benchmarking methods should inherit from this class and implement
    the required abstract methods.
    
    Parameters
    ----------
    method_name : str
        Name of the method (e.g., 'COSG', 'Wilcoxon', 'NS-Forest')
    params : dict, optional
        Method-specific parameters
        
    Attributes
    ----------
    method_name : str
        Name of the method
    params : dict
        Method parameters
    results_ : dict
        Stores results after calling find_markers()
    """
    
    def __init__(
        self,
        method_name: str,
        params: Optional[Dict] = None
    ):
        self.method_name = method_name
        self.params = params if params is not None else {}
        self.results_ = {}
        
        logger.info(f"Initialized {self.method_name} with params: {self.params}")
    
    @abstractmethod
    def find_markers(
        self,
        adata: AnnData,
        query_key: str,
        query_clusters: Optional[List[str]] = None,
        background_key: Optional[str] = None,
        n_genes: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """
        Find marker genes for query clusters relative to background.
        
        This is the main method that each implementation must define.
        
        Parameters
        ----------
        adata : AnnData
            Merged AnnData object containing both query and background cells
        query_key : str
            Key in adata.obs that identifies query clusters
            Example: 'cell_type' where values are 'VIP', 'SST', etc.
        query_clusters : list of str, optional
            Specific clusters to find markers for. If None, use all clusters.
        background_key : str, optional
            Key in adata.obs to identify background cells.
            Example: 'dataset' where background cells have value 'background'
            If None, use all non-query cells as background.
        n_genes : int, default=50
            Number of top marker genes to return per cluster
        **kwargs
            Method-specific additional arguments
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'cluster': query cluster name
            - 'gene': gene name
            - 'score': marker score (higher = better marker)
            - 'rank': rank within cluster (1 = best)
            - Additional method-specific columns
            
        Notes
        -----
        The returned DataFrame should be sorted by cluster and rank.
        """
        pass
    
    def _validate_inputs(
        self,
        adata: AnnData,
        query_key: str,
        background_key: Optional[str] = None
    ) -> None:
        """
        Validate input parameters.
        
        Parameters
        ----------
        adata : AnnData
            Input data
        query_key : str
            Query cluster key
        background_key : str, optional
            Background key
            
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        if query_key not in adata.obs.columns:
            raise ValueError(f"query_key '{query_key}' not found in adata.obs")
        
        if background_key is not None and background_key not in adata.obs.columns:
            raise ValueError(f"background_key '{background_key}' not found in adata.obs")
        
        logger.info(f"Input validation passed for {self.method_name}")
    
    def _get_query_background_split(
        self,
        adata: AnnData,
        query_key: str,
        cluster: str,
        background_key: Optional[str] = None
    ) -> tuple[AnnData, AnnData]:
        """
        Split data into query cluster cells and background cells.
        
        Parameters
        ----------
        adata : AnnData
            Full dataset
        query_key : str
            Key identifying clusters
        cluster : str
            Specific cluster to extract
        background_key : str, optional
            If provided, use this to identify background cells
            If None, use all cells not in the query cluster
            
        Returns
        -------
        query_adata : AnnData
            Cells from the query cluster
        background_adata : AnnData
            Background cells
        """
        # Get query cluster cells
        query_mask = adata.obs[query_key] == cluster
        query_adata = adata[query_mask].copy()
        
        # Get background cells
        if background_key is not None:
            # Use explicit background key
            background_mask = adata.obs[background_key] == 'background'
            background_adata = adata[background_mask].copy()
        else:
            # Use all non-query cells
            background_mask = ~query_mask
            background_adata = adata[background_mask].copy()
        
        logger.info(f"Split for cluster '{cluster}': "
                   f"{query_adata.n_obs} query cells, "
                   f"{background_adata.n_obs} background cells")
        
        return query_adata, background_adata
    
    def _format_results(
        self,
        results_dict: Dict[str, pd.DataFrame],
        n_genes: int
    ) -> pd.DataFrame:
        """
        Format method-specific results into standardized DataFrame.
        
        Parameters
        ----------
        results_dict : dict
            Dictionary mapping cluster names to DataFrames with marker info
        n_genes : int
            Number of top genes to keep per cluster
            
        Returns
        -------
        pd.DataFrame
            Standardized results DataFrame
        """
        all_results = []
        
        for cluster, df in results_dict.items():
            # Ensure required columns exist
            if 'gene' not in df.columns:
                raise ValueError(f"Results for cluster {cluster} missing 'gene' column")
            if 'score' not in df.columns:
                raise ValueError(f"Results for cluster {cluster} missing 'score' column")
            
            # Add cluster column
            df = df.copy()
            df['cluster'] = cluster
            
            # Sort by score (descending) and add rank
            df = df.sort_values('score', ascending=False)
            df['rank'] = range(1, len(df) + 1)
            
            # Keep top n_genes
            df = df.head(n_genes)
            
            all_results.append(df)
        
        # Concatenate all clusters
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns: cluster, gene, score, rank, then others
        cols = ['cluster', 'gene', 'score', 'rank']
        other_cols = [c for c in final_df.columns if c not in cols]
        final_df = final_df[cols + other_cols]
        
        return final_df
    
    def score_gene_list(
        self,
        adata: AnnData,
        gene_list: List[str],
        query_key: str,
        cluster: str,
        background_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Score a predefined list of genes for a specific cluster.
        
        This is useful for evaluating known marker genes or comparing
        across methods.
        
        Parameters
        ----------
        adata : AnnData
            Merged dataset
        gene_list : list of str
            Genes to score
        query_key : str
            Query cluster key
        cluster : str
            Specific cluster
        background_key : str, optional
            Background identifier
            
        Returns
        -------
        pd.DataFrame
            Scores for the provided genes
        """
        # This is optional - subclasses can override if they have
        # efficient ways to score specific genes
        raise NotImplementedError(
            f"{self.method_name} does not implement score_gene_list(). "
            "Run find_markers() instead."
        )
    
    def get_params(self) -> Dict:
        """Get method parameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> None:
        """Set method parameters."""
        self.params.update(params)
        logger.info(f"Updated {self.method_name} params: {params}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method_name='{self.method_name}', params={self.params})"


# Example of how a concrete implementation would look:
class ExampleMethod(BaseBenchmarkMethod):
    """
    Example implementation showing the interface.
    
    This is just for illustration - not a real method.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(method_name="Example", params=params)
    
    def find_markers(
        self,
        adata: AnnData,
        query_key: str,
        query_clusters: Optional[List[str]] = None,
        background_key: Optional[str] = None,
        n_genes: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """Find markers using the example method."""
        # Validate inputs
        self._validate_inputs(adata, query_key, background_key)
        
        # Get clusters to process
        if query_clusters is None:
            query_clusters = adata.obs[query_key].unique()
        
        results_dict = {}
        
        # Process each cluster
        for cluster in query_clusters:
            logger.info(f"Processing cluster: {cluster}")
            
            # Get query and background split
            query_adata, background_adata = self._get_query_background_split(
                adata, query_key, cluster, background_key
            )
            
            # === METHOD-SPECIFIC LOGIC GOES HERE ===
            # Example: simple fold-change calculation
            query_mean = np.array(query_adata.X.mean(axis=0)).flatten()
            background_mean = np.array(background_adata.X.mean(axis=0)).flatten()
            
            # Avoid division by zero
            background_mean = np.maximum(background_mean, 1e-10)
            fold_change = query_mean / background_mean
            
            # Create results DataFrame
            cluster_results = pd.DataFrame({
                'gene': adata.var_names,
                'score': fold_change,
                'query_mean': query_mean,
                'background_mean': background_mean
            })
            # === END METHOD-SPECIFIC LOGIC ===
            
            results_dict[cluster] = cluster_results
        
        # Format and return results
        final_results = self._format_results(results_dict, n_genes)
        self.results_ = final_results
        
        return final_results


# if __name__ == "__main__":
#     # Quick test of the base class design
#     print("Base class design loaded successfully!")
    
#     # Show what the interface looks like
#     example = ExampleMethod(params={'threshold': 0.5})
#     print(example)
#     print(f"Parameters: {example.get_params()}")