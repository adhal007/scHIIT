"""
Merge query (Seattle AD) and reference AnnData when reference has no metadata.

Reference data: Just a raw count matrix (163412 × 22298)
Query data: Full Seattle AD metadata
"""

import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

def fix_query_var_names(adata):
    neuronal_tier1 = adata.copy()
    # Save original Ensembl IDs
    neuronal_tier1.var['ensembl_id'] = neuronal_tier1.var_names.copy()
    
    # Convert categorical to string
    neuronal_tier1.var['feature_name'] = neuronal_tier1.var['feature_name'].astype(str)
    
    # Set var_names to gene symbols
    neuronal_tier1.var_names = neuronal_tier1.var['feature_name'].values
    
    # Make unique (handles the 28 duplicates)
    neuronal_tier1.var_names_make_unique()
    
    print("\nNew var_names (first 5):")
    print(neuronal_tier1.var_names[:5])
    
    print("\nDuplicate genes that got suffix:")
    duplicated_genes = neuronal_tier1.var_names[neuronal_tier1.var_names.str.contains('-[0-9]$', regex=True)]
    print(duplicated_genes[:10])
    return neuronal_tier1
    
def merge_query_reference_simple(
    query_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    query_cluster_key: str = 'Subclass',
    batch_key: str = 'dataset'
) -> ad.AnnData:
    """
    Merge query and reference when reference has NO metadata columns.
    
    This is the simple case - reference is just a background matrix.
    
    Parameters
    ----------
    query_adata : AnnData
        Seattle AD with full metadata (17937 × 36412)
    reference_adata : AnnData
        Tabula Sapiens with no .obs columns (163412 × 22298)
    query_cluster_key : str
        Column in query for cell types ('Class', 'Subclass', or 'Supertype')
    batch_key : str
        Name for new column identifying query vs reference
        
    Returns
    -------
    merged_adata : AnnData
        Merged object ready for benchmarking
    """
    
    print("="*60)
    print("MERGING QUERY + REFERENCE (Reference has no metadata)")
    print("="*60)
    
    # Step 1: Find common genes
    print("\nStep 1: Finding common genes...")
    common_genes = query_adata.var_names.intersection(reference_adata.var_names)
    print(f"  Query genes:     {query_adata.n_vars:,}")
    print(f"  Reference genes: {reference_adata.n_vars:,}")
    print(f"  Common genes:    {len(common_genes):,}")
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found! Check gene naming (symbols vs ENSEMBL IDs)")
    
    # Subset to common genes
    query_subset = query_adata[:, common_genes].copy()
    reference_subset = reference_adata[:, common_genes].copy()
    
    # Step 2: Add minimal metadata to reference
    print("\nStep 2: Adding metadata...")
    
    # Query: add dataset identifier
    query_subset.obs[batch_key] = 'query'
    query_subset.obs['cluster'] = query_subset.obs[query_cluster_key].astype(str)
    
    # Reference: create minimal metadata
    reference_subset.obs[batch_key] = 'reference'
    reference_subset.obs['cluster'] = 'background'  # All reference cells = background
    
    # Add the query cluster key column to reference (with 'background' values)
    reference_subset.obs[query_cluster_key] = 'background'
    
    print(f"  Query clusters: {query_subset.obs['cluster'].nunique()}")
    print(f"  Reference: all labeled as 'background'")
    
    # Step 3: Handle query-specific metadata columns
    print("\nStep 3: Preserving query metadata...")
    query_only_cols = set(query_subset.obs.columns) - set(reference_subset.obs.columns)
    print(f"  Query-specific columns: {len(query_only_cols)}")
    
    # Add these columns to reference with NA values
    for col in query_only_cols:
        dtype = query_subset.obs[col].dtype
        if dtype == 'object' or dtype.name == 'category':
            reference_subset.obs[col] = pd.Categorical(['NA'] * reference_subset.n_obs)
        elif dtype in ['float64', 'float32', 'int64', 'int32']:
            reference_subset.obs[col] = np.nan
        else:
            reference_subset.obs[col] = 'NA'
    
    # Step 4: Ensure both have same var columns
    print("\nStep 4: Aligning gene metadata...")
    if query_subset.var.shape[1] > 0:
        # Query has gene metadata, add to reference
        for col in query_subset.var.columns:
            if col not in reference_subset.var.columns:
                reference_subset.var[col] = 'NA'
    
    # Step 5: Merge
    print("\nStep 5: Concatenating datasets...")
    merged_adata = ad.concat(
        [query_subset, reference_subset],
        axis=0,
        join='outer',
        label='batch_source',
        keys=['query', 'reference'],
        index_unique='-',
        merge='same'
    )
    
    # Verify batch_key exists
    if batch_key not in merged_adata.obs.columns:
        merged_adata.obs[batch_key] = merged_adata.obs['batch_source']
    
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Total cells:      {merged_adata.n_obs:,}")
    print(f"Total genes:      {merged_adata.n_vars:,}")
    print(f"Query cells:      {(merged_adata.obs[batch_key] == 'query').sum():,}")
    print(f"Reference cells:  {(merged_adata.obs[batch_key] == 'reference').sum():,}")
    
    print("\nQuery cluster distribution:")
    query_clusters = merged_adata[merged_adata.obs[batch_key] == 'query'].obs['cluster'].value_counts()
    for cluster, count in query_clusters.items():
        print(f"  {cluster}: {count:,} cells")
    
    return merged_adata


def verify_merge(merged_adata: ad.AnnData, batch_key: str = 'dataset'):
    """
    Verify the merge worked correctly.
    """
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check basic structure
    print("\n1. Basic structure:")
    print(f"   Total cells: {merged_adata.n_obs:,}")
    print(f"   Total genes: {merged_adata.n_vars:,}")
    print(f"   Sparse matrix: {issparse(merged_adata.X)}")
    
    # Check dataset split
    print("\n2. Dataset distribution:")
    print(merged_adata.obs[batch_key].value_counts())
    
    # Check cluster key
    print("\n3. Cluster key:")
    print(f"   Column exists: {'cluster' in merged_adata.obs.columns}")
    print(f"   Unique clusters: {merged_adata.obs['cluster'].nunique()}")
    print(f"   Query clusters: {merged_adata[merged_adata.obs[batch_key]=='query'].obs['cluster'].nunique()}")
    
    # Check query metadata preservation
    print("\n4. Query metadata preservation:")
    query_specific_cols = ['Class', 'Subclass', 'Supertype', 'Cognitive status', 'Braak stage']
    for col in query_specific_cols:
        if col in merged_adata.obs.columns:
            n_query_with_data = merged_adata[
                merged_adata.obs[batch_key] == 'query'
            ].obs[col].notna().sum()
            print(f"   {col}: {n_query_with_data:,} query cells with data ✓")
        else:
            print(f"   {col}: NOT FOUND ✗")
    
    # Check for any issues
    print("\n5. Sanity checks:")
    
    # All reference cells should be 'background' cluster
    ref_clusters = merged_adata[merged_adata.obs[batch_key] == 'reference'].obs['cluster'].unique()
    if len(ref_clusters) == 1 and ref_clusters[0] == 'background':
        print("   Reference cells all labeled 'background': ✓")
    else:
        print(f"   WARNING: Reference has {len(ref_clusters)} clusters: {ref_clusters}")
    
    # No NaN in critical columns
    critical_cols = [batch_key, 'cluster']
    all_good = True
    for col in critical_cols:
        if merged_adata.obs[col].isna().any():
            print(f"   WARNING: NaN values in '{col}' column")
            all_good = False
    
    if all_good:
        print("   No NaN in critical columns: ✓")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)


