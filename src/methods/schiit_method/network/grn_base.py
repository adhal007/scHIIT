"""
Stage III: Core TF Identification using Network Analysis and Strongly Connected Components
===========================================================================================

This module identifies the core set of transcription factors by:
1. Starting with unique TFs from Stage I/II
2. Adding non-unique TFs that connect unique TFs via ChIP-seq
3. Building a regulatory network
4. Extracting strongly connected components (SCCs)
5. Selecting the largest/most relevant SCC as the core TF set

Author: Custom implementation for TF identity detection pipeline
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import warnings


class StageIIICoreIdentifier:
    """
    Identifies core TFs by combining unique TFs with connecting non-unique TFs
    and extracting strongly connected components.
    """
    def __init__(
        self,
        unique_tfs: List[str],
        high_exp_tfs: List[str],
        chipseq_file: str,
        unique_genes: Optional[List[str]] = None,
        high_exp_genes: Optional[List[str]] = None,
        use_collectri: bool = True,
        verbose: bool = True,
        # NEW parameters for connecting TF selection
        jsd_scores: Optional[Dict] = None,                    # JSD scores from Stage I/II
        connecting_selection_method: str = 'top_percent',     # 'top_percent', 'top_n', 'all'
        connecting_top_percent: float = 1.0,                  # Default: top 1%
        connecting_top_n: int = None,                         # Alternative: top N
    ):
        """
        Initialize Stage III Core Identifier.
        
        Args:
            unique_tfs: List of unique TFs from Stage I/II (core_filtered_tfs for TFs)
            high_exp_tfs: List of all highly expressed TFs from Stage I/II (high_exp_tfs)
            chipseq_file: Path to ChIP-seq data file (TSV with TF, Gene columns)
            unique_genes: Optional list of unique genes from Stage I/II
            high_exp_genes: Optional list of all highly expressed genes from Stage I/II
            verbose: Print progress information
        """
        self.unique_tfs = set(unique_tfs)
        self.high_exp_tfs = set(high_exp_tfs)
        self.chipseq_file = chipseq_file
        self.unique_genes = set(unique_genes) if unique_genes else set()
        self.high_exp_genes = set(high_exp_genes) if high_exp_genes else set()
        self.verbose = verbose
        self.use_collectri = use_collectri
        self.jsd_scores = jsd_scores
        self.connecting_selection_method = connecting_selection_method
        self.connecting_top_percent = connecting_top_percent
        self.connecting_top_n = connecting_top_n
        # Results storage
        self.pkn_network = None
        self.connecting_tfs = set()
        self.core_tf_set = set()
        self.regulatory_network = None
        self.sccs = []
        self.largest_scc = []
        self.results = {}
        
    def load_chipseq_pkn(self) -> pd.DataFrame:
        """Load literature network."""
        if self.verbose:
            print("\n" + "="*80)
            print("STEP 1: LOAD LITERATURE NETWORK")
            print("="*80)
        
        edges = []
        
        # ChIP-seq
        if self.chipseq_file:
            if self.verbose:
                print(f"\nLoading ChIP-seq: {self.chipseq_file}")
            
            chipseq = pd.read_csv(self.chipseq_file, sep='\t', header=None)
            pkn_chip = chipseq.iloc[:, [0, 1]].copy()
            pkn_chip.columns = ['source', 'target']
            pkn_chip['evidence'] = 'chipseq'
            
            if self.verbose:
                print(f"  ChIP-seq edges: {len(pkn_chip)}")
            edges.append(pkn_chip)

        # CollecTRI
        if self.use_collectri:
            if self.verbose:
                print("\nLoading CollecTRI...")
            try:
                import decoupler as dc
                    
                #handle both old and new decoupler API
                try:
                    ct = dc.get_collectri(organism='human')  #new API (>=1.4)
                except AttributeError:
                    ct = dc.op.collectri(organism='human')   #old API
                    
                #handle both old and new dataframe structures
                if 'sign_decision' in ct.columns:
                    #old format: filter by sign_decision
                    collectri = ct[ct['sign_decision'] == 'default activation']
                elif 'weight' in ct.columns:
                    #new format: filter by positive weight
                    collectri = ct[ct['weight'] > 0]
                else:
                    #fallback: use all edges
                    collectri = ct
                    
                pkn_collectri = collectri[['source', 'target']].copy()
                pkn_collectri['evidence'] = 'collectri'
                    
                if self.verbose:
                    print(f"  CollecTRI edges: {len(pkn_collectri)}")
                edges.append(pkn_collectri)
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ CollecTRI failed: {e}")       

        
        if len(edges) == 0:
            raise ValueError("No literature network!")
        
        literature_net = pd.concat(edges, ignore_index=True)
        literature_net = literature_net[literature_net['source'] != literature_net['target']]
        literature_net = literature_net.groupby(['source', 'target'])['evidence'].apply(
            lambda x: ','.join(x.unique())
        ).reset_index()
        
        self.pkn_network  = literature_net
        
        if self.verbose:
            print(f"\n✓ Literature network: {len(literature_net)} edges")
        
        return literature_net
    
    # REPLACE the identify_connecting_tfs method (lines 124-187) with this enhanced version:
    def identify_connecting_tfs(self) -> Set[str]:
        """
        Identify non-unique TFs that connect unique TFs, optionally filtered by score.
        
        A non-unique TF is selected if:
        - It has at least one unique TF as a source (regulating it)
        - It has at least one unique TF as a target (it regulates)
        - [NEW] It ranks in top X% by JSD score (if filtering enabled)
        
        Returns:
            Set of connecting TF names
        """
        if self.verbose:
            print("\n=== Identifying Connecting TFs ===")
        
        if self.pkn_network is None:
            raise RuntimeError("Must load PKN first using load_chipseq_pkn()")
        
        # Get non-unique TFs (in high_exp but not in unique)
        non_unique_tfs = self.high_exp_tfs - self.unique_tfs
        
        if self.verbose:
            print(f"  Unique TFs: {len(self.unique_tfs)}")
            print(f"  Non-unique TFs to evaluate: {len(non_unique_tfs)}")
        
        # ========================================================================
        # NEW: Pre-filter non-unique TFs by score if scores are available
        # ========================================================================
        candidate_tfs = non_unique_tfs
        
        if hasattr(self, 'jsd_scores') and self.jsd_scores:
            # Get connecting selection parameters (with defaults)
            selection_method = getattr(self, 'connecting_selection_method', 'top_percent')
            top_percent = getattr(self, 'connecting_top_percent', 1.0)
            top_n = getattr(self, 'connecting_top_n', None)
            
            # Rank non-unique TFs by score
            scored_tfs = []
            for tf in non_unique_tfs:
                if tf in self.jsd_scores:
                    score = self.jsd_scores[tf]
                    # Handle tuple (directional) vs float (simple)
                    if isinstance(score, tuple):
                        score_val = abs(score[2])  # Use abs(signed_specificity)
                    else:
                        score_val = score
                    scored_tfs.append((tf, score_val))
            
            if scored_tfs and selection_method != 'all':
                # Determine sort order
                is_directional = isinstance(list(self.jsd_scores.values())[0], tuple)
                reverse = is_directional  # directional: high=better, simple: low=better
                
                # Sort
                ranked = sorted(scored_tfs, key=lambda x: x[1], reverse=reverse)
                
                # Select top subset
                if selection_method == 'top_percent':
                    n_select = max(1, int(len(ranked) * top_percent / 100))
                    candidate_tfs = set([tf for tf, _ in ranked[:n_select]])
                    if self.verbose:
                        print(f"  Pre-filtered to top {top_percent}% = {n_select} non-unique TFs by score")
                
                elif selection_method == 'top_n' and top_n:
                    n_select = min(top_n, len(ranked))
                    candidate_tfs = set([tf for tf, _ in ranked[:n_select]])
                    if self.verbose:
                        print(f"  Pre-filtered to top {n_select} non-unique TFs by score")
        
        # ========================================================================
        # Original connectivity check (on filtered candidates)
        # ========================================================================
        connecting = set()
        tf_connections = defaultdict(lambda: {'sources': set(), 'targets': set()})
        
        for tf in candidate_tfs:
            # Find unique TFs that regulate this TF (sources)
            sources = self.pkn_network[
                (self.pkn_network['target'] == tf) &
                (self.pkn_network['source'].isin(self.unique_tfs))
            ]['source'].unique()
            
            # Find unique TFs that this TF regulates (targets)
            targets = self.pkn_network[
                (self.pkn_network['source'] == tf) &
                (self.pkn_network['target'].isin(self.unique_tfs))
            ]['target'].unique()
            
            # If has at least one source AND one target unique TF, include it
            if len(sources) > 0 and len(targets) > 0:
                connecting.add(tf)
                tf_connections[tf]['sources'] = set(sources)
                tf_connections[tf]['targets'] = set(targets)
        
        self.connecting_tfs = connecting
        self.tf_connections = tf_connections
        
        if self.verbose:
            print(f"  Found {len(connecting)} connecting TFs (after connectivity check)")
            
            # Show examples
            if len(connecting) > 0:
                print(f"\n  Example connections (first 3):")
                for i, tf in enumerate(list(connecting)[:3]):
                    conn = tf_connections[tf]
                    print(f"    {tf}:")
                    print(f"      Sources: {list(conn['sources'])[:3]}")
                    print(f"      Targets: {list(conn['targets'])[:3]}")
        
        return connecting
    
    def build_regulatory_network(self) -> nx.DiGraph:
        """
        Build directed regulatory network from:
        - Unique TFs
        - Connecting TFs
        - ChIP-seq edges between them
        
        Returns:
            NetworkX DiGraph representing the regulatory network
        """
        if self.verbose:
            print("\n=== Building Regulatory Network ===")
        
        # Core TF set = unique TFs + connecting TFs
        core_set = self.unique_tfs | self.connecting_tfs
        
        if self.verbose:
            print(f"  Core set size: {len(core_set)}")
            print(f"    - Unique TFs: {len(self.unique_tfs)}")
            print(f"    - Connecting TFs: {len(self.connecting_tfs)}")
        
        # Build network
        G = nx.DiGraph()
        G.add_nodes_from(core_set)
        
        # Add edges from ChIP-seq within the core set
        edges_added = 0
        for _, row in self.pkn_network.iterrows():
            source, target = row['source'], row['target']
            
            # Only add edges between nodes in the core set
            if source in core_set and target in core_set:
                G.add_edge(source, target)
                edges_added += 1
        
        self.regulatory_network = G
        
        if self.verbose:
            print(f"  Network statistics:")
            print(f"    Nodes: {G.number_of_nodes()}")
            print(f"    Edges: {G.number_of_edges()}")
            print(f"    Density: {nx.density(G):.4f}")
            
            # Check connectivity
            if G.number_of_nodes() > 0:
                wcc = list(nx.weakly_connected_components(G))
                print(f"    Weakly connected components: {len(wcc)}")
                if len(wcc) > 0:
                    print(f"    Largest WCC size: {len(max(wcc, key=len))}")
        
        return G
    
    def extract_strongly_connected_components(self) -> List[Set[str]]:
        """
        Extract strongly connected components from the regulatory network.
        
        SCCs represent groups of TFs that form regulatory cycles/feedback loops.
        
        Returns:
            List of SCCs, sorted by size (largest first)
        """
        if self.verbose:
            print("\n=== Extracting Strongly Connected Components ===")
        
        if self.regulatory_network is None:
            raise RuntimeError("Must build regulatory network first")
        
        # Get SCCs
        sccs = list(nx.strongly_connected_components(self.regulatory_network))
        
        # Sort by size (largest first)
        sccs_sorted = sorted(sccs, key=len, reverse=True)
        
        self.sccs = sccs_sorted
        
        if self.verbose:
            print(f"  Found {len(sccs_sorted)} SCCs")
            
            # Show size distribution
            if len(sccs_sorted) > 0:
                sizes = [len(scc) for scc in sccs_sorted]
                print(f"  SCC sizes: {sizes[:10]}")  # Show first 10
                
                # Show largest SCC details
                largest = sccs_sorted[0]
                print(f"\n  Largest SCC:")
                print(f"    Size: {len(largest)}")
                print(f"    Members: {list(largest)[:10]}")  # Show first 10
                
                # Count unique vs connecting TFs in largest SCC
                unique_in_largest = len(largest & self.unique_tfs)
                connecting_in_largest = len(largest & self.connecting_tfs)
                print(f"    Unique TFs: {unique_in_largest}")
                print(f"    Connecting TFs: {connecting_in_largest}")
        
        return sccs_sorted
    
    def select_core_tfs(
        self,
        method: str = 'largest',
        min_scc_size: int = 2,
        top_n_sccs: Optional[int] = None
    ) -> Set[str]:
        """
        Select final core TF set from SCCs.
        
        Args:
            method: Selection method
                - 'largest': Use largest SCC only
                - 'top_n': Use union of top N SCCs
                - 'all_above_threshold': Use all SCCs above min_scc_size
            min_scc_size: Minimum size for an SCC to be considered (default: 2)
            top_n_sccs: For 'top_n' method, number of SCCs to include
        
        Returns:
            Set of core TF names
        """
        if self.verbose:
            print(f"\n=== Selecting Core TFs (method: {method}) ===")
        
        if not self.sccs:
            raise RuntimeError("Must extract SCCs first")
        
        # Filter by minimum size
        valid_sccs = [scc for scc in self.sccs if len(scc) >= min_scc_size]
        
        if len(valid_sccs) == 0:
            warnings.warn(
                f"No SCCs found with size >= {min_scc_size}. "
                f"Using all unique TFs as core."
            )
            core = self.unique_tfs
        else:
            if method == 'largest':
                core = valid_sccs[0]
                self.largest_scc = list(core)
                
            elif method == 'top_n':
                if top_n_sccs is None:
                    raise ValueError("Must specify top_n_sccs for 'top_n' method")
                n = min(top_n_sccs, len(valid_sccs))
                core = set().union(*valid_sccs[:n])
                self.largest_scc = list(valid_sccs[0])
                
            elif method == 'all_above_threshold':
                core = set().union(*valid_sccs)
                self.largest_scc = list(valid_sccs[0])
                
            else:
                raise ValueError(f"Unknown method: {method}")
        
        self.core_tf_set = core
        
        if self.verbose:
            print(f"  Core TF set size: {len(core)}")
            print(f"  Composition:")
            unique_in_core = len(core & self.unique_tfs)
            connecting_in_core = len(core & self.connecting_tfs)
            print(f"    Unique TFs: {unique_in_core} ({100*unique_in_core/len(core):.1f}%)")
            print(f"    Connecting TFs: {connecting_in_core} ({100*connecting_in_core/len(core):.1f}%)")
        
        return core
    
    def run_full_pipeline(
        self,
        selection_method: str = 'largest',
        min_scc_size: int = 2,
        top_n_sccs: Optional[int] = None
    ) -> Dict:
        """
        Run complete Stage III pipeline.
        
        Args:
            selection_method: How to select core TFs from SCCs
            min_scc_size: Minimum SCC size to consider
            top_n_sccs: For 'top_n' method
        
        Returns:
            Dictionary with results:
                - 'core_tfs': Final core TF set
                - 'unique_tfs': Original unique TFs
                - 'connecting_tfs': Non-unique TFs that connect unique TFs
                - 'network': Regulatory network (DiGraph)
                - 'sccs': All strongly connected components
                - 'largest_scc': The largest SCC
                - 'statistics': Various statistics
        """
        if self.verbose:
            print("\n" + "="*70)
            print("STAGE III: CORE TF IDENTIFICATION")
            print("="*70)
        
        # Step 1: Load ChIP-seq PKN
        self.load_chipseq_pkn()
        
        # Step 2: Identify connecting TFs
        self.identify_connecting_tfs()
        
        # Step 3: Build regulatory network
        self.build_regulatory_network()
        
        # Step 4: Extract SCCs
        self.extract_strongly_connected_components()
        
        # Step 5: Select core TFs
        self.select_core_tfs(
            method=selection_method,
            min_scc_size=min_scc_size,
            top_n_sccs=top_n_sccs
        )
        
        # Compile results
        self.results = {
            'core_tfs': list(self.core_tf_set),
            'unique_tfs': list(self.unique_tfs),
            'connecting_tfs': list(self.connecting_tfs),
            'network': self.regulatory_network,
            'sccs': [list(scc) for scc in self.sccs],
            'largest_scc': self.largest_scc,
            'statistics': {
                'n_unique_tfs': len(self.unique_tfs),
                'n_connecting_tfs': len(self.connecting_tfs),
                'n_core_tfs': len(self.core_tf_set),
                'n_sccs': len(self.sccs),
                'largest_scc_size': len(self.largest_scc) if self.largest_scc else 0,
                'network_nodes': self.regulatory_network.number_of_nodes(),
                'network_edges': self.regulatory_network.number_of_edges(),
                'network_density': nx.density(self.regulatory_network),
                'unique_in_core': len(self.core_tf_set & self.unique_tfs),
                'connecting_in_core': len(self.core_tf_set & self.connecting_tfs),
            }
        }
        
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print comprehensive summary of Stage III results."""
        print("\n" + "="*70)
        print("STAGE III SUMMARY")
        print("="*70)
        
        stats = self.results['statistics']
        
        print("\nInput:")
        print(f"  Unique TFs (Stage I/II): {stats['n_unique_tfs']}")
        print(f"  Total highly expressed TFs: {len(self.high_exp_tfs)}")
        
        print("\nNetwork Analysis:")
        print(f"  Connecting TFs identified: {stats['n_connecting_tfs']}")
        print(f"  Network nodes: {stats['network_nodes']}")
        print(f"  Network edges: {stats['network_edges']}")
        print(f"  Network density: {stats['network_density']:.4f}")
        
        print("\nStrongly Connected Components:")
        print(f"  Total SCCs found: {stats['n_sccs']}")
        print(f"  Largest SCC size: {stats['largest_scc_size']}")
        if len(self.sccs) > 1:
            scc_sizes = [len(scc) for scc in self.sccs[:5]]
            print(f"  Top 5 SCC sizes: {scc_sizes}")
        
        print("\nFinal Core TF Set:")
        print(f"  Total core TFs: {stats['n_core_tfs']}")
        print(f"    - From unique TFs: {stats['unique_in_core']} "
              f"({100*stats['unique_in_core']/stats['n_core_tfs']:.1f}%)")
        print(f"    - From connecting TFs: {stats['connecting_in_core']} "
              f"({100*stats['connecting_in_core']/stats['n_core_tfs']:.1f}%)")
        
        print("\n" + "="*70)
    
    def export_results(self, output_dir: str):
        """
        Export results to files.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Core TFs
        core_df = pd.DataFrame({
            'TF': list(self.core_tf_set),
            'Type': ['unique' if tf in self.unique_tfs else 'connecting' 
                    for tf in self.core_tf_set]
        })
        core_df.to_csv(f"{output_dir}/core_tfs.csv", index=False)
        
        # Connecting TF details
        if self.connecting_tfs:
            conn_data = []
            for tf in self.connecting_tfs:
                conn = self.tf_connections[tf]
                conn_data.append({
                    'TF': tf,
                    'n_source_unique_tfs': len(conn['sources']),
                    'n_target_unique_tfs': len(conn['targets']),
                    'source_unique_tfs': ','.join(conn['sources']),
                    'target_unique_tfs': ','.join(conn['targets']),
                    'in_core': tf in self.core_tf_set
                })
            conn_df = pd.DataFrame(conn_data)
            conn_df.to_csv(f"{output_dir}/connecting_tfs_details.csv", index=False)
        
        # SCC information
        scc_data = []
        for i, scc in enumerate(self.sccs):
            scc_data.append({
                'SCC_id': i,
                'size': len(scc),
                'members': ','.join(scc),
                'n_unique': len(set(scc) & self.unique_tfs),
                'n_connecting': len(set(scc) & self.connecting_tfs)
            })
        scc_df = pd.DataFrame(scc_data)
        scc_df.to_csv(f"{output_dir}/sccs.csv", index=False)
        
        # Network edges
        if self.regulatory_network:
            edges = nx.to_pandas_edgelist(self.regulatory_network)
            edges.to_csv(f"{output_dir}/regulatory_network_edges.csv", index=False)
        
        # Statistics summary
        stats_df = pd.DataFrame([self.results['statistics']])
        stats_df.to_csv(f"{output_dir}/statistics.csv", index=False)
        
        if self.verbose:
            print(f"\nResults exported to: {output_dir}/")


def run_stage_iii(
    tf_results: Dict,
    gene_results: Dict,
    chipseq_file: str,
    selection_method: str = 'largest',
    min_scc_size: int = 2,
    output_dir: Optional[str] = None,
    high_exp_tfs: List[str] = None,
    verbose: bool = True,
    # NEW parameters
    connecting_selection_method: str = 'top_percent',
    connecting_top_percent: float = 1.0,
    connecting_top_n: int = None
) -> Dict:
    """
    Convenience function to run Stage III on results from Stage I/II.
    
    Args:
        tf_results: Results dict from FastHierarchicalOIFilter for TFs
        gene_results: Results dict from FastHierarchicalOIFilter for genes
        chipseq_file: Path to ChIP-seq data
        selection_method: 'largest', 'top_n', or 'all_above_threshold'
        min_scc_size: Minimum SCC size
        output_dir: Optional directory to export results
        verbose: Print progress
    
    Returns:
        Dictionary with Stage III results
    
    Example:
        >>> results_stage3 = run_stage_iii(
        ...     tf_results=results['GABAergic_neuron'],
        ...     gene_results=results_genes['GABAergic_neuron'],
        ...     chipseq_file='/path/to/chipseq.tsv'
        ... )
        >>> core_tfs = results_stage3['core_tfs']
    """
    # Extract JSD scores
    jsd_scores = tf_results.get('jsd_scores', None)
    if high_exp_tfs is None or len(high_exp_tfs) == 0:
        identifier = StageIIICoreIdentifier(
            unique_tfs=tf_results['identity_tfs'],
            high_exp_tfs=tf_results['high_exp_tfs'],
            chipseq_file=chipseq_file,
            unique_genes=gene_results.get('identity_tfs', []),
            high_exp_genes=gene_results.get('high_exp_tfs', []),
            verbose=verbose,
            # NEW
            jsd_scores=jsd_scores,
            connecting_selection_method=connecting_selection_method,
            connecting_top_percent=connecting_top_percent,
            connecting_top_n=connecting_top_n
        )
    else:
        identifier = StageIIICoreIdentifier(
            unique_tfs=tf_results['identity_tfs'],
            high_exp_tfs=high_exp_tfs,
            chipseq_file=chipseq_file,
            unique_genes=gene_results.get('identity_tfs', []),
            high_exp_genes=gene_results.get('high_exp_tfs', []),
            verbose=verbose,
            # NEW
            jsd_scores=jsd_scores,
            connecting_selection_method=connecting_selection_method,
            connecting_top_percent=connecting_top_percent,
            connecting_top_n=connecting_top_n
        )
            
    results = identifier.run_full_pipeline(
        selection_method=selection_method,
        min_scc_size=min_scc_size
    )
    
    if output_dir:
        identifier.export_results(output_dir)
    
    return results
