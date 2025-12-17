#!/usr/bin/env python3
"""
Minimal script to build CellxGene metadata repository for specific organs
Run on HPC with sbatch
"""

import scanpy as sc 
import pandas as pd
import sys 
import os 
project_root = os.path.abspath(os.path.join(os.path.dirname('src'), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import src.io.cellxgene_pp_utils as cxg_utils
from importlib import reload 
reload(cxg_utils)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "/mnt/lscratch/users/adhal/SingleCellUtils/data/scBackGroundRag/sc"

# Organ to tissue mapping (matching CellxGene Census tissue_general terms)
ORGAN_CONFIG = {
    # Neurons
    'brain': {
        'tissues': ['brain', 'nervous system'],
        'datasets': 100
    },
    
    # Cardiac
    'heart': {
        'tissues': ['heart', 'cardiovascular system'],
        'datasets': 50
    },
    
    # Hepatocyte
    'liver': {
        'tissues': ['liver'],
        'datasets': 50
    },
    
    # Kidney
    'kidney': {
        'tissues': ['kidney', 'renal system'],
        'datasets': 50
    },
    
    # B cell, Macrophage, Monocyte
    'blood': {
        'tissues': ['blood', 'blood vessel', 'hematopoietic system', 'bone marrow', 'spleen'],
        'datasets': 100
    },
    
    # Skeletal muscle
    'skeletal_muscle': {
        'tissues': ['skeletal muscle', 'muscle organ'],
        'datasets': 50
    },
    
    # Endothelial cell (can be from various tissues)
    'vasculature': {
        'tissues': ['blood vessel', 'artery', 'vein', 'cardiovascular system'],
        'datasets': 50
    },
}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    print("="*80)
    print("BUILDING CELLXGENE METADATA REPOSITORY - OPTIMIZED")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Organs to process: {len(ORGAN_CONFIG)}")
    
    total_datasets = sum(config['datasets'] for config in ORGAN_CONFIG.values())
    print(f"Target total datasets: ~{total_datasets}")
    print("="*80)
    print()
    
    # Initialize builder
    builder = cxg_utils.MetadataBuilder(output_dir=OUTPUT_DIR)
    
    # Build repository with parallelization
    organs = list(ORGAN_CONFIG.keys())
    
    # Use the maximum datasets across organs for consistency
    max_datasets = max(config['datasets'] for config in ORGAN_CONFIG.values())
    
    builder.build_repository(
        organs=organs,
        datasets_per_organ=max_datasets,
        use_multiprocessing=True,
        n_processes=min(8, len(organs))  # Match your SLURM cpus-per-task
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("✓ METADATA BUILD COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    
    # Print summary
    import json
    from pathlib import Path
    
    metadata_files = list(Path(OUTPUT_DIR).glob("*.json"))
    metadata_files = [f for f in metadata_files if f.stem != 'manifest']
    
    print(f"\nCreated {len(metadata_files)} dataset metadata files")
    
    # Count by organ
    organ_counts = {}
    for mf in metadata_files:
        organ = mf.stem.split('_')[0]
        organ_counts[organ] = organ_counts.get(organ, 0) + 1
    
    print("\nDatasets per organ:")
    for organ, count in sorted(organ_counts.items()):
        print(f"  {organ}: {count}")