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

This example demonstrates the end-to-end scHIIT pipeline for gene regulatory network (GRN) inference from single-cell data.

3.1 Setup and Load Packages
----------------------------

.. code-block:: python

    import scanpy as sc
    import pandas as pd
    import numpy as np
    import sys
    import os
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
    ref_data_path = "path/to/reference_nuclei.h5ad"
    ref_adata = sc.read_h5ad(ref_data_path)
    
    # Load query data (your dataset of interest)
    query_data_path = "path/to/query_data.h5ad"
    query_adata = sc.read_h5ad(query_data_path)
    
    # Important: Ensure adata.var_names are in GENE NAME format, not ENSEMBL ID

3.3 Harmonize and Merge Query + Reference
------------------------------------------

Merge query and reference datasets while preserving metadata:

.. code-block:: python

    import src.io.seattle_ad.query_ref_merge as query_ref_merge
    
    # Merge datasets with proper harmonization
    merged_adata = query_ref_merge.merge_query_reference(
        query_adata=query_adata,
        ref_adata=ref_adata,
        cluster_key="cluster_name"  # Column with cell type annotations
    )
    
    # The merge function:
    # 1. Finds common genes between query and reference
    # 2. Adds metadata distinguishing query vs reference cells
    # 3. Preserves query-specific metadata (e.g., disease status, donor info)
    # 4. Aligns gene metadata
    # 5. Creates unified dataset with proper labeling
    
    print(f"Total cells: {merged_adata.n_obs:,}")
    print(f"Total genes: {merged_adata.n_vars:,}")
    print(f"Query cells: {(merged_adata.obs['dataset'] == 'query').sum():,}")
    print(f"Reference cells: {(merged_adata.obs['dataset'] == 'reference').sum():,}")

3.4 Initialize JSD Filter
--------------------------

Create the Jensen-Shannon Divergence filter for identifying TF-target relationships:

.. code-block:: python

    # Initialize JSD filter with merged data
    jsd_filter = schiit_main.JSDFilter(
        adata=merged_adata,
        cluster_key="cluster_name",  # Column with cell type labels
        background_cluster="background"  # Label for reference cells
    )
    
    # The JSD filter computes:
    # - Gene expression distributions per cluster
    # - Jensen-Shannon divergence between query and background
    # - TF-specific and target-specific filters

3.5 Compute TF-Target Interactions
-----------------------------------

Run the filter to identify significant TF-target pairs:

.. code-block:: python

    # Define your transcription factors of interest
    tfs_of_interest = ["FOXP2", "MEF2C", "NEUROD6", "TBR1", "SATB2"]
    
    # Compute JSD-filtered interactions
    filtered_interactions = jsd_filter.filter_tf_targets(
        tfs=tfs_of_interest,
        clusters=["Neuronal: Glutamatergic", "Neuronal: GABAergic"],
        jsd_threshold=0.1,  # Minimum JSD threshold
        expr_threshold=0.5  # Minimum expression threshold
    )
    
    # Results include:
    # - TF name
    # - Target gene
    # - Cluster
    # - JSD score
    # - Expression levels
    # - Statistical significance

3.6 Build Gene Regulatory Network
----------------------------------

Construct the GRN from filtered interactions:

.. code-block:: python

    # Initialize GRN builder
    grn_builder = schiit_grn.GRNBuilder()
    
    # Build network from filtered interactions
    grn = grn_builder.build_network(
        interactions=filtered_interactions,
        cluster="Neuronal: Glutamatergic"
    )
    
    # Analyze network properties
    network_stats = grn_builder.compute_network_statistics(grn)
    print(f"Nodes: {network_stats['n_nodes']}")
    print(f"Edges: {network_stats['n_edges']}")
    print(f"Density: {network_stats['density']:.4f}")
    print(f"Average degree: {network_stats['avg_degree']:.2f}")

3.7 Visualize the Network
--------------------------

Create visualizations of the inferred GRN:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Create network visualization
    fig, ax = plt.subplots(figsize=(15, 15))
    
    grn_builder.plot_network(
        grn=grn,
        cluster="Neuronal: Glutamatergic",
        layout="spring",  # or "kamada_kawai", "circular"
        node_size_by="degree",
        color_by="module",
        show_labels=True,
        ax=ax
    )
    
    plt.tight_layout()
    plt.savefig("grn_glutamatergic.pdf", dpi=300, bbox_inches='tight')
    plt.show()

3.8 Export Results
-------------------

Save your results for downstream analysis:

.. code-block:: python

    # Export filtered interactions to CSV
    filtered_interactions.to_csv(
        "schiit_filtered_interactions.csv",
        index=False
    )
    
    # Export network in various formats
    grn_builder.export_network(
        grn=grn,
        output_path="schiit_grn",
        formats=["graphml", "edgelist", "gml"]
    )
    
    # Save network statistics
    with open("network_stats.json", "w") as f:
        import json
        json.dump(network_stats, f, indent=2)

3.9 Compare Across Clusters
----------------------------

Compare GRNs across different cell type clusters:

.. code-block:: python

    clusters = ["Neuronal: Glutamatergic", "Neuronal: GABAergic"]
    
    # Build networks for each cluster
    grn_dict = {}
    for cluster in clusters:
        interactions = jsd_filter.filter_tf_targets(
            tfs=tfs_of_interest,
            clusters=[cluster],
            jsd_threshold=0.1
        )
        grn_dict[cluster] = grn_builder.build_network(interactions, cluster)
    
    # Compare network properties
    comparison = grn_builder.compare_networks(grn_dict)
    
    # Visualize comparison
    fig, axes = plt.subplots(1, len(clusters), figsize=(20, 8))
    for ax, (cluster, grn) in zip(axes, grn_dict.items()):
        grn_builder.plot_network(grn, cluster, ax=ax)
    plt.tight_layout()
    plt.savefig("grn_comparison.pdf", dpi=300, bbox_inches='tight')

3.10 Advanced: Custom Filtering Parameters
-------------------------------------------

Fine-tune the filtering parameters for your specific use case:

.. code-block:: python

    # Custom filter with multiple thresholds
    filtered_interactions = jsd_filter.filter_tf_targets(
        tfs=tfs_of_interest,
        clusters=["Neuronal: Glutamatergic"],
        jsd_threshold=0.15,        # Higher JSD threshold
        expr_threshold=0.5,        # Minimum expression
        specificity_threshold=0.3, # Cluster specificity
        top_n_per_tf=100,          # Keep top 100 targets per TF
        remove_housekeeping=True   # Filter out housekeeping genes
    )
    
    # Apply additional biological filters
    filtered_interactions = jsd_filter.apply_biological_filters(
        interactions=filtered_interactions,
        known_interactions_db="path/to/known_interactions.csv",
        require_known=False,  # Don't require known interactions
        penalize_unknown=True # But penalize completely unknown pairs
    )

.. tip::
   The scHIIT method uses Jensen-Shannon Divergence to identify TF-target pairs that show differential expression patterns between your query cells and background reference cells. Higher JSD scores indicate stronger specificity to your cell type of interest.

Summary
=======

These examples demonstrate the three main capabilities of the SCHIIT toolkit:

1. **Data Retrieval**: Efficient querying of the CELLxGENE Census database with caching
2. **Cell Type Harmonization**: Intelligent ontology-based cell type annotation standardization
3. **GRN Inference**: Complete pipeline for identifying gene regulatory networks in single-cell data

For more detailed information on each module, see the :doc:`api/modules` documentation.

.. seealso::
   - :doc:`installation` - Setup instructions
   - :doc:`quickstart` - Quick introduction to basic usage
   - :doc:`api/modules` - Complete API reference
   - :doc:`references` - Citation information