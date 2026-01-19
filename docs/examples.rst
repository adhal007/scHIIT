Examples
========

This section provides practical examples of using the SCHIIT toolkit for single-cell analysis, including data retrieval from CELLxGENE Census, cell type ontology harmonization, and the complete scHIIT pipeline for gene regulatory network inference.

.. note::
   The SCHIIT package is not yet published on PyPI. To use these examples, clone the repository and ensure your environment is set up according to the :doc:`installation` guide.

1. CELLxGENE Census Data Retrieval
===================================

The SCHIIT toolkit provides utilities for querying and retrieving single-cell data from the CELLxGENE Census database with efficient caching.

1.1 Basic Setup and Caching
----------------------------

Initialize the CELLxGENE utilities with caching to speed up repeated queries:

.. code-block:: python

    import src.io.cellxgene_pp_utils as cxg_utils
    
    # Initialize with default filter
    utils = cxg_utils.CellxgenePpUtils(organism='homo_sapiens')
    
    # First call - fetches from CELLxGENE (slow)
    print("First call (cache miss):")
    tissues = utils.get_tissue_options()
    
    # Second call - returns cached result (fast)
    print("\nSecond call (cache hit):")
    tissues = utils.get_tissue_options()
    
    # Check cache statistics
    utils.print_cache_stats()
    # Output:
    # 📊 Cache Statistics
    # ------------------------------
    #   Cached entries: 1
    #   Hits: 1
    #   Misses: 1
    #   Hit rate: 50.0%

1.2 Preloading Common Filters
------------------------------

Preload commonly used filter options at startup for better performance:

.. code-block:: python

    # Preload common filter options at startup
    utils.preload_cache()
    # Output:
    # Preloading cache for 7 columns...
    #   Loading: suspension_type... ✓
    #   Loading: disease... ✓
    #   Loading: tissue_general... ✓
    #   Loading: tissue... ✓
    #   Loading: assay... ✓
    #   Loading: sex... ✓
    #   Loading: development_stage... ✓
    
    # Now all subsequent calls are fast
    suspension = utils.get_suspension_type_options()  # Cache hit
    assays = utils.get_assay_options()                # Cache hit

1.3 Querying Multiple Tissues
------------------------------

Query data for multiple tissues with various filter combinations:

.. code-block:: python

    # Example 1: Use default filter (disease == 'normal' and is_primary_data == True)
    query1 = utils.get_multiple_tissues(['lung', 'heart'])
    
    # Example 2: Add additional filters to the default
    query2 = utils.get_multiple_tissues(
        ['lung', 'heart'],
        additional_filters=["sex == 'female'", "development_stage == 'adult'"]
    )
    
    # Example 3: Use a completely custom filter (no default)
    query3 = utils.get_multiple_tissues(
        ['lung'],
        custom_filter="disease == 'COVID-19' and is_primary_data == True"
    )
    
    # Example 4: Update default filter for subsequent queries
    utils.update_default_filter("disease == 'COVID-19' and is_primary_data == True")
    query4 = utils.get_multiple_tissues(['blood'])

1.4 Cache Management
--------------------

Manage the cache for optimal performance:

.. code-block:: python

    # Force refresh (bypass cache)
    tissues_fresh = utils.get_tissue_options(use_cache=False)
    
    # Clear cache when needed
    utils.clear_cache()
    
    # Updating default filter clears cache automatically
    utils.update_default_filter("disease == 'COVID-19' and is_primary_data == True")

.. note::
   The cache is automatically cleared when you update the default filter to ensure consistency in your queries.

2. Cell Type Ontology Harmonization
====================================

SCHIIT includes tools for harmonizing cell type annotations using the Cell Ontology, with intelligent filtering of functional vs. biological parent relationships.

2.1 Initialize the Ontology Visualizer
---------------------------------------

.. code-block:: python

    from src.io.ontology_utils import OntologyVisualizer
    
    # Load Cell Ontology from OBO file
    visualizer = OntologyVisualizer(obo_path="path/to/cl.obo")
    # Output:
    # Loading Cell Ontology...
    # ✓ Loaded 2,845 terms

2.2 Query Cell Type Context
----------------------------

Visualize the ontology context for any cell type to understand its hierarchical relationships:

.. code-block:: python

    # Visualize context for a specific cell type
    visualizer.visualize_query_context("astrocyte", show_definitions=True)
    
    # You can also query by Cell Ontology ID
    visualizer.visualize_query_context("CL:0000127")  # astrocyte
    
    # The output shows:
    # - The term definition
    # - All parent terms (biological vs functional)
    # - Sibling cell types through each parent
    # - Children (more specific) cell types

2.3 Understanding Biological vs Functional Parents
---------------------------------------------------

The ontology classifier distinguishes between biological lineage parents and functional classification parents:

.. code-block:: python

    # Get only biological parents (excludes functional classifications)
    bio_parents = visualizer.get_biological_parents("CL:0000540")  # neuron
    
    # This filters out functional parents like:
    # - "electrically responsive cell"
    # - "secretory cell"
    # - "signaling cell"
    
    # And keeps biological lineage parents like:
    # - "central nervous system neuron"
    # - "peripheral nervous system neuron"

2.4 Find Sibling Cell Types
----------------------------

Discover related cell types at the same hierarchical level:

.. code-block:: python

    # Get all siblings through biological parents
    siblings = visualizer.get_siblings("CL:0000127")  # astrocyte
    
    # Get siblings through a specific parent
    siblings = visualizer.get_siblings(
        "CL:0000127",  # astrocyte
        through_parent="CL:0000125"  # glial cell
    )
    
    # Each sibling includes:
    # - Cell type ID and name
    # - Parent through which sibling relationship exists
    # - Parent ID

2.5 Harmonize Cell Types to Broader Categories
-----------------------------------------------

.. code-block:: python

    from src.methods.schiit_method.celltype_harmonizer import CellTypeHarmonizer
    
    # Initialize harmonizer
    harmonizer = CellTypeHarmonizer(obo_path="path/to/cl.obo")
    
    # List of specific cell types to harmonize
    cell_types = [
        "sst GABAergic cortical interneuron",
        "vip GABAergic cortical interneuron",
        "pvalb GABAergic cortical interneuron",
        "mature astrocyte",
        "protoplasmic astrocyte",
        "oligodendrocyte precursor cell"
    ]
    
    # Harmonize to broader categories (2 levels up in hierarchy)
    harmonized = harmonizer.harmonize_types(cell_types, levels_up=2, verbose=True)
    
    # Results grouped by broad category:
    # GABAergic interneuron (3 types):
    #   - sst GABAergic cortical interneuron
    #   - vip GABAergic cortical interneuron
    #   - pvalb GABAergic cortical interneuron
    # 
    # astrocyte (2 types):
    #   - mature astrocyte
    #   - protoplasmic astrocyte
    #
    # glial cell (1 type):
    #   - oligodendrocyte precursor cell

2.6 Preview Harmonization at Different Levels
----------------------------------------------

.. code-block:: python

    # Preview what harmonization would look like at different hierarchy levels
    harmonizer.preview_harmonization(cell_types, max_levels=3)
    
    # This shows how cell types would be grouped at:
    # - Level 0: original cell types (no harmonization)
    # - Level 1: one level up in hierarchy
    # - Level 2: two levels up in hierarchy
    # - Level 3: three levels up in hierarchy

.. warning::
   The harmonizer automatically filters out functional parent classifications to ensure biological coherence. Cell types like neurons will NOT be mapped to "electrically responsive cell" but will follow their biological lineage.


3. Complete scHIIT Pipeline
============================

This example demonstrates the end-to-end scHIIT pipeline for gene regulatory network (GRN) inference from single-cell data using the three-stage approach.

3.1 Setup and Load Packages
----------------------------

.. code-block:: python

    import scanpy as sc
    import pandas as pd
    import numpy as np
    import sys
    import os
    from UniProtMapper import ProtMapper
    from importlib import reload
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname('src'), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import scHIIT modules
    import src.methods.schiit_method.tf_filters.base_filter as schiit_base
    import src.methods.schiit_method.tf_filters.jsd_filter as schiit_main
    import src.methods.schiit_method.network.grn_base as schiit_grn
    
    # Reload for development
    reload(schiit_main)
    reload(schiit_grn)

3.2 Load Query and Reference Data
----------------------------------

The scHIIT pipeline requires both query data (your dataset of interest) and reference data (background cells):

.. code-block:: python

    # Load reference data (single nuclei background)
    ref_data_path = "/path/to/ref_nuclei.h5ad"
    ref_adata = sc.read_h5ad(ref_data_path)
    
    # Load query data (your dataset of interest)
    query_data_path = "/path/to/query_data.h5ad"
    query_adata = sc.read_h5ad(query_data_path)

.. important::
   Ensure ``adata.var_names`` are in GENE NAME format (e.g., "FOXP2"), NOT ENSEMBL ID format (e.g., "ENSG00000128573").

3.3 Fix Query Variable Names and Merge Datasets
------------------------------------------------

Fix variable names and merge query with reference data:

.. code-block:: python

    import src.io.seattle_ad.query_ref_merge as query_ref_merge
    
    # Fix query variable names (converts ENSEMBL IDs to gene names if needed)
    query_adata_fixed = query_ref_merge.fix_query_var_names(query_adata)
    
    # Merge query and reference datasets
    merged_adata = query_ref_merge.merge_query_reference_simple(
        query_adata=query_adata_fixed,
        reference_adata=ref_adata,
        query_cluster_key='Class',  # Options: 'Class', 'Subclass', 'Supertype'
        batch_key='dataset'
    )
    
    # Verify the merge
    query_ref_merge.verify_merge(merged_adata, batch_key='dataset')
    
    print("\nMerged data ready for benchmarking! ✓")
    
    # Optional: Save merged data
    # merged_adata.write('merged_data.h5ad')
    
    # Clean up
    del query_adata_fixed

The merge function performs the following:

- Finds common genes between query and reference
- Adds metadata distinguishing query vs reference cells
- Preserves query-specific metadata (e.g., disease status, donor info)
- Aligns gene metadata
- Creates unified dataset with proper labeling

3.4 Create Memory-Safe Subset
------------------------------

For large datasets, create a memory-safe subset by subsampling background cells:

.. code-block:: python

    import numpy as np
    import gc
    from scipy.sparse import csr_matrix, issparse
    
    def create_subset(adata, target_ct, max_background=20000):
        """
        Create a subset with target cells and sampled background cells.
        
        Parameters
        ----------
        adata : AnnData
            Full merged dataset
        target_ct : str
            Target cell type (e.g., 'Neuronal: GABAergic')
        max_background : int
            Maximum number of background cells to keep
        
        Returns
        -------
        AnnData
            Subsetted dataset
        """
        target_mask = adata.obs['Class'] == target_ct
        background_mask = adata.obs['Class'] == 'background'
        
        target_idx = np.where(target_mask)[0]
        background_idx = np.where(background_mask)[0]
        
        # Subsample background if too large
        if len(background_idx) > max_background:
            np.random.seed(42)
            background_idx = np.random.choice(
                background_idx, max_background, replace=False
            )
        
        keep_idx = np.concatenate([target_idx, background_idx])
        subset = adata[keep_idx, :].copy()
        
        # Ensure sparse matrix format
        if not issparse(subset.X):
            subset.X = csr_matrix(subset.X)
        
        return subset
    
    # Create subset for your target cell type
    ct = 'Neuronal: GABAergic'
    adata_safe = create_subset(merged_adata, ct, max_background=50000)

3.5 Stage I + II: Core TF Filtering Pipeline
---------------------------------------------

Run Stages I and II to identify core transcription factors using bidirectional GJSD:

.. code-block:: python

    # Load TF list
    tf_df = pd.read_csv(
        '/path/to/Homo_sapiens_TF.txt', 
        sep='\t'
    )
    tf_name_list = tf_df['Symbol'].to_list()
    
    # Initialize results storage
    results = {}
    stageI_II_tfs = {}
    
    # Set method
    method = 'bidirectional_gjsd'
    
    # Initialize Core Filter Pipeline
    neuro_top_tf = schiit_main.CoreFilterOnlyPipeline(
        adata=adata_safe,
        tf_list=tf_name_list,
        target_cell_type=ct,
        background_cell_type='background',
        cell_type_key='Class',
        chipseq_file=None,
        verbose=True,
        scgx_sig_file='/path/to/sig_frames/Neuronal: GABAergic_sig_tbl.txt',
        main_filter='high_and_unique',
        jsd_method=method,
        top_n_high=None,
        top_jsd_pc=None,
        top_n_jsd=1000,
        expr_method='scgx',
        identity_top_percent=10.0,  # Top 10% for identity
    )
    
    # Run the pipeline
    neuro_top_tf.run()
    
    # Store results
    results[method] = neuro_top_tf.results
    stageI_II_tfs[method] = neuro_top_tf.results['core_filtered_tfs']
    
    # Clean up memory
    del adata_safe
    gc.collect()

**Key Parameters:**

- ``main_filter``: Filtering strategy ('high_and_unique', 'unique_only', etc.)
- ``jsd_method``: Method for computing divergence ('bidirectional_gjsd', 'forward_gjsd', etc.)
- ``top_n_jsd``: Number of top genes to keep based on JSD score
- ``identity_top_percent``: Top percentage for identity scoring
- ``expr_method``: Expression method ('scgx', 'mean', 'median')

3.6 Examine Directional GJSD Results
-------------------------------------

Analyze the directional generalized Jensen-Shannon Divergence scores:

.. code-block:: python

    # Get full directional DataFrame
    df = neuro_top_tf.get_directional_scores_df()
    
    print("\n=== Directional GJSD Results ===")
    print(df.head(10))
    
    # Filter for TARGET-SPECIFIC genes
    target_specific = df[
        (df['signed_specificity'] > 0) &           # Positive direction
        (df['specificity_target'] > 0.65) &        # High confidence
        (df['gjsd_score'] > 1.0)                   # Minimum divergence
    ].sort_values('signed_specificity', ascending=False)
    
    print(f"\nTarget-specific genes: {len(target_specific)}")
    print(target_specific[[
        'gjsd_score', 'direction', 'specificity_target',
        'mean_target', 'mean_other'
    ]].head(20))
    
    # Filter for BACKGROUND-SPECIFIC genes
    background_specific = df[
        (df['signed_specificity'] < 0) &
        (df['specificity_other'] > 0.65) &
        (df['gjsd_score'] > 1.0)
    ].sort_values('signed_specificity', ascending=True)
    
    print(f"\nBackground-specific genes: {len(background_specific)}")

3.7 Visualize Top Target-Specific Genes
----------------------------------------

Create a tracks plot to visualize expression of top target-specific genes:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Set font sizes
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Plot top 20 target-specific genes
    ax = sc.pl.tracksplot(
        merged_adata,
        target_specific.index.values[:20],
        groupby="batch_source",
        dendrogram=False,
        figsize=(15, 8),
        show=False
    )
    plt.show()

3.8 Stage III: Build Core Transcriptional Network
--------------------------------------------------

Run Stage III to construct the core gene regulatory network using ChIP-seq data:

.. code-block:: python

    import src.methods.schiit_method.network.grn_base as schiit_grn
    reload(schiit_grn)
    
    # Path to ChIP-seq data
    chip_seq_file = '/path/to/Chip_removed_overlapped_peaks.tsv'
    
    # Run Stage III
    stage3_results = schiit_grn.run_stage_iii(
        tf_results=results['bidirectional_gjsd'],
        gene_results=results['bidirectional_gjsd'],
        chipseq_file=chip_seq_file,
        selection_method='largest',
        min_scc_size=2,
        verbose=True,
        connecting_top_percent=100  # Use top 1% for connecting TFs
    )

**Stage III performs:**

1. Filters TFs and genes based on Stages I and II
2. Constructs TF-target interactions using ChIP-seq evidence
3. Builds directed graph of regulatory relationships
4. Identifies strongly connected components (SCCs)
5. Selects core transcriptional network

**Key Parameters:**

- ``selection_method``: How to select core network ('largest', 'highest_degree', etc.)
- ``min_scc_size``: Minimum size for strongly connected components
- ``connecting_top_percent``: Percentage of top connecting TFs to include

3.9 Visualize Core Network
---------------------------

Plot the final core transcriptional network:

.. code-block:: python

    import src.io.plotting_utils as putils
    
    # Plot core network
    putils.plot_core_network(stage3_results)

The visualization shows:

- Nodes representing TFs and target genes
- Edges representing regulatory relationships
- Node colors indicating modules or functional groups
- Node sizes proportional to connectivity or importance

3.10 Optional: Run Pipeline for All Genes
------------------------------------------

You can also run the pipeline for all genes (not just TFs) to identify regulated targets:

.. code-block:: python

    # Create gene list (excluding TFs)
    gene_list = list(set(adata_safe.var_names) - set(tf_name_list))
    
    # Initialize pipeline for genes
    results_genes = {}
    stageI_II_genes = {}
    
    neuro_top_genes = schiit_main.CoreFilterOnlyPipeline(
        adata=adata_safe,
        tf_list=gene_list,  # Use gene list instead of TF list
        target_cell_type=ct,
        background_cell_type='background',
        cell_type_key='Class',
        chipseq_file=None,
        verbose=True,
        scgx_sig_file='/path/to/sig_frames/Neuronal: GABAergic_sig_tbl.txt',
        main_filter='unique_only',
        jsd_method=method,
        top_n_high=700,
        top_jsd_pc=None,
        top_n_jsd=30,
        expr_method='scgx',
    )
    
    # Run pipeline
    neuro_top_genes.run()
    results_genes[method] = neuro_top_genes.results
    stageI_II_genes[method] = neuro_top_genes.results['core_filtered_tfs']
    
    # Clean up
    del adata_safe
    gc.collect()

.. tip::
   The three-stage scHIIT pipeline progressively refines the network: Stage I identifies highly expressed genes, Stage II applies bidirectional GJSD filtering to find cell-type-specific genes, and Stage III constructs the core regulatory network using ChIP-seq evidence.

Summary
=======

These examples demonstrate the three main capabilities of the SCHIIT toolkit:

1. **Data Retrieval**: Efficient querying of the CELLxGENE Census database with caching
2. **Cell Type Harmonization**: Intelligent ontology-based cell type annotation standardization
3. **GRN Inference**: Complete pipeline for identifying gene regulatory networks in single-cell data

For more detailed information on each module, see the :doc:`api/modules` documentation.

.. seealso::
   - :doc:`getting_started`` - Setup instructions
   - :doc:`quickstart` - Quick introduction to basic usage
..    - :doc:`api/modules` - Complete API reference
..    - :doc:`references` - Citation information
