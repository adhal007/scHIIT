
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import issparse
from scipy.stats import rankdata
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, Counter
import warnings
import decoupler as dc
warnings.filterwarnings('ignore')


class StageIIICoreIdentifier:
    """
    Find identity core using InfoMap community detection.
    
    Strategy:
    1. Get all PKN edges where source OR target is identity TF
    2. Filter non-identity TFs: must be in high_exp_tfs
    3. Filter edges by correlation (TF-TF co-expression in target cells)
    4. Run InfoMap community detection
    5. Core = community with most identity TFs
    """
    
    def __init__(
        self,
        identity_tfs: List[str],
        high_exp_tfs: List[str],
        chipseq_file: str = None,
        pkn: pd.DataFrame = None,
        # Correlation filter (required for reasonable scaffold size)
        adata = None,
        target_cell_type: str = None,
        cell_type_key: str = 'cell_type',
        corr_threshold: float = 0.3,
        corr_method: str = 'spearman',
        use_collectri:bool = True, 
        collectri_organism:str='human',
        # InfoMap parameters
        min_community_size: int = 3,
        verbose: bool = True
    ):
        self.identity_tfs = set(identity_tfs)
        self.high_exp_tfs = set(high_exp_tfs)
        self.chipseq_file = chipseq_file
        self.pkn = pkn
        self.adata = adata
        self.target_cell_type = target_cell_type
        self.cell_type_key = cell_type_key
        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.min_community_size = min_community_size
        self.use_collectri = use_collectri
        self.collectri_organism = collectri_organism
        self.verbose = verbose
        
        # Results
        self.scaffold_tfs = set()
        self.scaffold_network = None
        self.correlation_matrix = None
        self.communities = []
        self.identity_community = set()
        self.results = {}
    
    def load_pkn(self) -> pd.DataFrame:
        """Load PKN and filter to identity TF ↔ identity TF edges only."""
        if self.verbose:
            print("\n" + "="*60)
            print("STEP 1: LOAD PKN (Identity-Identity Only)")
            print("="*60)
        
        if self.pkn is None:
            if self.chipseq_file is None:
                raise ValueError("Must provide either pkn or chipseq_file")
            
            if self.verbose:
                print(f"  Loading from {self.chipseq_file}...")
            
            chipseq = pd.read_csv(self.chipseq_file, sep='\t')
            
            if 'TF' in chipseq.columns and 'Gene' in chipseq.columns:
                self.pkn = chipseq[['TF', 'Gene']].copy()
                self.pkn.columns = ['source', 'target']
            else:
                self.pkn = chipseq[['source', 'target']].copy()
            
            self.pkn = self.pkn.drop_duplicates()
            self.pkn['evidence'] = 'chip'
        # CollecTRI
        if self.use_collectri:
            if self.verbose:
                print("\nLoading CollecTRI...")
                #handle both old and new decoupler API
                try:
                    ct = dc.get_collectri(organism=self.use_collectri)  #new API (>=1.4)
                except AttributeError:
                    ct = dc.op.collectri(organism=self.collectri_organism)   #old API
                    
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

                print('\nMerging CHIP and Collectri')
                self.pkn = pd.concat([self.pkn, pkn_collectri], axis=1)

        if self.verbose:
            print(f"  Total PKN edges: {len(self.pkn)}")
            print(f"  Identity TFs: {len(self.identity_tfs)}")
        
        # Filter to identity ↔ identity edges only (both source AND target must be identity TFs)
        mask = (
            self.pkn['source'].isin(self.identity_tfs) & 
            self.pkn['target'].isin(self.identity_tfs)
        )
        self.identity_pkn = self.pkn[mask].copy()
        
        # Get identity TFs that have at least one edge
        connected_identity = set(self.identity_pkn['source']) | set(self.identity_pkn['target'])
        isolated_identity = self.identity_tfs - connected_identity
        
        if self.verbose:
            print(f"  Identity ↔ Identity edges: {len(self.identity_pkn)}")
            print(f"  Connected identity TFs: {len(connected_identity)}/{len(self.identity_tfs)}")
            if isolated_identity:
                print(f"  Isolated identity TFs (no PKN edges): {len(isolated_identity)}")
        
        return self.identity_pkn
    
    def compute_correlations(self) -> Optional[pd.DataFrame]:
        """
        Compute correlations for identity TFs.
        Used for edge weighting, not filtering.
        """
        if self.adata is None or self.target_cell_type is None:
            if self.verbose:
                print("\n  ⚠ No adata/target_cell_type - skipping correlation")
            return None
        
        if self.verbose:
            print("\n" + "="*60)
            print("STEP 2: COMPUTE CORRELATIONS (for edge weighting)")
            print("="*60)
        
        # Get identity TFs from PKN
        tfs_in_pkn = set(self.identity_pkn['source']) | set(self.identity_pkn['target'])
        
        # Subset to target cells
        mask = self.adata.obs[self.cell_type_key] == self.target_cell_type
        adata_target = self.adata[mask]
        
        # Filter to TFs in expression data
        tfs_in_data = [tf for tf in tfs_in_pkn if tf in adata_target.var_names]
        
        if self.verbose:
            print(f"  Target cells: {adata_target.n_obs}")
            print(f"  Identity TFs in PKN: {len(tfs_in_pkn)}")
            print(f"  Identity TFs in expression data: {len(tfs_in_data)}")
        
        if len(tfs_in_data) < 2:
            return None
        
        # Get expression matrix
        X = adata_target[:, tfs_in_data].X
        if issparse(X):
            X = X.toarray()
        
        if self.verbose:
            print(f"  Computing {len(tfs_in_data)} × {len(tfs_in_data)} correlations...")
        
        # Vectorized Spearman: rank transform then Pearson
        if self.corr_method == 'spearman':
            X_ranked = np.apply_along_axis(rankdata, 0, X)
        else:
            X_ranked = X
        
        # Center and normalize
        X_centered = X_ranked - X_ranked.mean(axis=0)
        X_norm = X_centered / (X_centered.std(axis=0) + 1e-10)
        
        # Correlation matrix via matrix multiplication
        corr_matrix = (X_norm.T @ X_norm) / X_norm.shape[0]
        
        self.correlation_matrix = pd.DataFrame(
            corr_matrix,
            index=tfs_in_data,
            columns=tfs_in_data
        )
        
        if self.verbose:
            print(f"  ✓ Correlation matrix: {self.correlation_matrix.shape}")
        
        return self.correlation_matrix
    
    def build_network(self) -> nx.DiGraph:
        """
        Build network from identity ↔ identity PKN edges.
        No filtering needed - we trust the PKN for identity TFs.
        Optionally weight edges by correlation.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STEP 3: BUILD NETWORK")
            print("="*60)
        
        G = nx.DiGraph()
        
        # Add all identity-identity edges
        for _, row in self.identity_pkn.iterrows():
            src, tgt = row['source'], row['target']
            
            # Optionally add correlation as edge weight
            weight = 1.0
            if self.correlation_matrix is not None:
                if src in self.correlation_matrix.index and tgt in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[src, tgt]
                    if not np.isnan(corr):
                        weight = abs(corr)
            
            G.add_edge(src, tgt, weight=weight)
        
        self.scaffold_network = G
        self.scaffold_tfs = set(G.nodes())
        
        if self.verbose:
            print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"  All nodes are identity TFs")
            
            # Show isolated identity TFs (not in PKN)
            isolated = self.identity_tfs - self.scaffold_tfs
            if isolated:
                print(f"  Identity TFs not in network: {len(isolated)}")
        
        return G
    
    def run_infomap(self) -> List[Set[str]]:
        """
        Run InfoMap community detection.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STEP 4: INFOMAP COMMUNITY DETECTION")
            print("="*60)
        
        try:
            from infomap import Infomap
        except ImportError:
            if self.verbose:
                print("  ⚠ infomap not installed, falling back to Louvain")
            return self._run_louvain_fallback()
        
        # Create Infomap instance
        im = Infomap(silent=True, directed=True)
        
        # Map nodes to integers
        node_to_id = {node: i for i, node in enumerate(self.scaffold_network.nodes())}
        id_to_node = {i: node for node, i in node_to_id.items()}
        
        # Add edges
        for src, tgt, data in self.scaffold_network.edges(data=True):
            weight = data.get('weight', 1.0)
            im.add_link(node_to_id[src], node_to_id[tgt], weight)
        
        # Run InfoMap
        im.run()
        
        # Extract communities
        community_dict = defaultdict(set)
        for node_id, module_id in im.modules:
            community_dict[module_id].add(id_to_node[node_id])
        
        self.communities = [comm for comm in community_dict.values() 
                          if len(comm) >= self.min_community_size]
        
        # Sort by size
        self.communities.sort(key=len, reverse=True)
        
        if self.verbose:
            print(f"  Found {len(self.communities)} communities (size >= {self.min_community_size})")
            for i, comm in enumerate(self.communities[:5]):
                identity_in_comm = len(comm & self.identity_tfs)
                print(f"    Community {i+1}: {len(comm)} nodes ({identity_in_comm} identity TFs)")
        
        return self.communities
    
    def _run_louvain_fallback(self) -> List[Set[str]]:
        """Fallback to Louvain if InfoMap not available."""
        try:
            import community as community_louvain
        except ImportError:
            # Try networkx community
            from networkx.algorithms import community as nx_community
            
            # Convert to undirected for Louvain
            G_undirected = self.scaffold_network.to_undirected()
            
            # Run Louvain
            communities_generator = nx_community.louvain_communities(G_undirected)
            self.communities = [comm for comm in communities_generator 
                              if len(comm) >= self.min_community_size]
            self.communities.sort(key=len, reverse=True)
            
            if self.verbose:
                print(f"  Found {len(self.communities)} communities (Louvain fallback)")
            
            return self.communities
        
        # Use python-louvain
        G_undirected = self.scaffold_network.to_undirected()
        partition = community_louvain.best_partition(G_undirected)
        
        community_dict = defaultdict(set)
        for node, comm_id in partition.items():
            community_dict[comm_id].add(node)
        
        self.communities = [comm for comm in community_dict.values() 
                          if len(comm) >= self.min_community_size]
        self.communities.sort(key=len, reverse=True)
        
        if self.verbose:
            print(f"  Found {len(self.communities)} communities (Louvain fallback)")
        
        return self.communities
    
    def identify_core_community(self) -> Set[str]:
        """
        Find community containing most identity TFs.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STEP 5: IDENTIFY CORE COMMUNITY")
            print("="*60)
        
        if not self.communities:
            if self.verbose:
                print("  ⚠ No communities found")
            return set()
        
        # Score each community by identity TF content
        community_scores = []
        for i, comm in enumerate(self.communities):
            identity_in_comm = comm & self.identity_tfs
            n_identity = len(identity_in_comm)
            pct_identity = 100 * n_identity / len(comm) if comm else 0
            pct_of_all_identity = 100 * n_identity / len(self.identity_tfs) if self.identity_tfs else 0
            
            community_scores.append({
                'index': i,
                'size': len(comm),
                'n_identity': n_identity,
                'pct_identity': pct_identity,
                'pct_of_all_identity': pct_of_all_identity,
                'community': comm,
                'identity_tfs': identity_in_comm
            })
        
        # Sort by number of identity TFs (primary) and pct identity (secondary)
        community_scores.sort(key=lambda x: (x['n_identity'], x['pct_identity']), reverse=True)
        
        if self.verbose:
            print("\n  Community ranking by identity TF content:")
            for cs in community_scores[:5]:
                print(f"    Community {cs['index']+1}: {cs['n_identity']} identity TFs "
                      f"({cs['pct_identity']:.1f}% of community, "
                      f"{cs['pct_of_all_identity']:.1f}% of all identity)")
        
        # Best community = most identity TFs
        best = community_scores[0]
        self.identity_community = best['community']
        
        if self.verbose:
            print(f"\n  ✓ Core community: {len(self.identity_community)} TFs")
            print(f"    Identity TFs: {sorted(best['identity_tfs'])}")
            
            # Non-identity TFs in core (these are the "bridging" TFs)
            bridging = self.identity_community - self.identity_tfs
            if bridging:
                print(f"    Bridging TFs: {sorted(bridging)}")
        
        return self.identity_community
    
    def classify_identity_tfs(self) -> Dict:
        """
        Classify identity TFs relative to core community.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STEP 6: CLASSIFY IDENTITY TFs")
            print("="*60)
        
        # Identity TFs in core
        core_identity = self.identity_community & self.identity_tfs
        
        # Identity TFs outside core
        outside_core = self.identity_tfs - core_identity
        
        # Check if outside TFs are in other communities
        in_other_community = set()
        isolated = set()
        
        for tf in outside_core:
            found = False
            for comm in self.communities:
                if tf in comm and comm != self.identity_community:
                    in_other_community.add(tf)
                    found = True
                    break
            if not found:
                isolated.add(tf)
        
        classification = {
            'core': sorted(list(core_identity)),
            'other_community': sorted(list(in_other_community)),
            'isolated': sorted(list(isolated)),
            'bridging_tfs': sorted(list(self.identity_community - self.identity_tfs))
        }
        
        if self.verbose:
            print(f"\n  Core community: {len(core_identity)} identity TFs")
            print(f"    {classification['core']}")
            print(f"  In other communities: {len(in_other_community)}")
            if in_other_community:
                print(f"    {classification['other_community']}")
            print(f"  Isolated: {len(isolated)}")
            if isolated:
                print(f"    {classification['isolated']}")
            print(f"  Bridging (non-identity in core): {len(classification['bridging_tfs'])}")
            if classification['bridging_tfs']:
                print(f"    {classification['bridging_tfs'][:10]}...")
        
        return classification
    
    def run(self) -> Dict:
        """Run full pipeline."""
        if self.verbose:
            print("\n" + "="*70)
            print("INFOMAP IDENTITY CORE DETECTION")
            print("="*70)
            print(f"\nInput: {len(self.identity_tfs)} identity TFs")
            print(f"Correlation threshold: {self.corr_threshold}")
        
        # Step 1: Load PKN
        self.load_pkn()
        
        # Step 2: Compute correlations (for edge filtering)
        self.compute_correlations()
        
        # Step 3: Build network (identity-anchored with correlation filter)
        self.build_network()
        
        # Step 4: Run InfoMap
        self.run_infomap()
        
        # Step 5: Find core community
        self.identify_core_community()
        
        # Step 6: Classify
        classification = self.classify_identity_tfs()
        
        # Map to grn_base.py terminology:
        # - core_scc → identity TFs in the core community (the "circuit")
        # - upstream_tfs → bridging TFs (non-identity TFs that connect identity TFs)
        # - downstream_tfs → identity TFs in other communities
        # - isolated_tfs → identity TFs not in any community
        
        core_scc = set(classification['core'])
        upstream_tfs = set(classification['bridging_tfs'])  # These bridge/regulate the core
        downstream_tfs = set(classification['other_community'])
        isolated_tfs = set(classification['isolated'])
        
        # Full identity core = core + downstream (all captured identity TFs)
        full_identity_core = core_scc | downstream_tfs
        
        # Conversion candidates = core identity TFs + bridging TFs
        conversion_candidates = core_scc | upstream_tfs
        
        # Build results (backward compatible with grn_base.py)
        self.results = {
            # Main outputs (backward compatible)
            'core_tfs': sorted(list(core_scc)),
            'identity_tfs': sorted(list(full_identity_core)),
            
            # Detailed classification
            'core_scc': sorted(list(core_scc)),
            'upstream_tfs': sorted(list(upstream_tfs)),  # Bridging TFs
            'downstream_tfs': sorted(list(downstream_tfs)),  # Other community TFs
            'isolated_tfs': sorted(list(isolated_tfs)),
            
            # Input tracking
            'identity_tfs_input': sorted(list(self.identity_tfs)),
            'high_exp_tfs': sorted(list(self.high_exp_tfs)),
            
            # For cell conversion
            'conversion_candidates': sorted(list(conversion_candidates)),
            
            # Network
            'network': self.scaffold_network,
            'sccs': [sorted(list(c)) for c in self.communities],  # Communities as "SCCs"
            'largest_scc': sorted(list(self.identity_community)),  # Core community
            
            # InfoMap-specific
            'core_community': sorted(list(self.identity_community)),
            'bridging_tfs': sorted(list(upstream_tfs)),  # Alias
            'communities': [sorted(list(c)) for c in self.communities],
            'n_communities': len(self.communities),
            
            # Path information (empty for InfoMap approach)
            'upstream_paths': {},
            'downstream_paths': {},
            
            # Correlation results
            'correlation_matrix': self.correlation_matrix,
            
            # Statistics
            'statistics': {
                'n_identity_input': len(self.identity_tfs),
                'n_high_exp_tfs': len(self.high_exp_tfs),
                'n_scaffold': len(self.scaffold_tfs),
                'n_core_scc': len(core_scc),
                'n_upstream': len(upstream_tfs),
                'n_downstream': len(downstream_tfs),
                'n_isolated': len(isolated_tfs),
                'n_identity_captured': len(full_identity_core),
                'pct_identity_captured': 100 * len(full_identity_core) / len(self.identity_tfs) if self.identity_tfs else 0,
                'n_conversion_candidates': len(conversion_candidates),
                'network_nodes': self.scaffold_network.number_of_nodes() if self.scaffold_network else 0,
                'network_edges': self.scaffold_network.number_of_edges() if self.scaffold_network else 0,
                'n_sccs': len(self.communities),
                'n_communities': len(self.communities),
                'n_core_community': len(self.identity_community),
                'n_bridging': len(upstream_tfs),
            }
        }
        
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print pipeline summary (matches grn_base.py format)."""
        print("\n" + "="*70)
        print("STAGE III SUMMARY (InfoMap)")
        print("="*70)
        
        stats = self.results['statistics']
        
        print(f"\nInput: {stats['n_identity_input']} identity TFs from Stage I/II")
        
        print(f"\nClassification of Identity TFs:")
        print(f"  ✓ Core (in main community): {stats['n_core_scc']}")
        if self.results['core_scc']:
            print(f"      {self.results['core_scc']}")
        
        print(f"  ↑ Bridging TFs (non-identity in core): {stats['n_upstream']}")
        if self.results['upstream_tfs']:
            tfs = self.results['upstream_tfs']
            if len(tfs) > 10:
                print(f"      {tfs[:10]} ... (+{len(tfs)-10} more)")
            else:
                print(f"      {tfs}")
        
        print(f"  ↓ Other communities: {stats['n_downstream']}")
        if self.results['downstream_tfs']:
            print(f"      {self.results['downstream_tfs']}")
        
        print(f"  ○ Isolated: {stats['n_isolated']}")
        if self.results['isolated_tfs']:
            print(f"      {self.results['isolated_tfs']}")
        
        print(f"\nIdentity TFs captured: {stats['n_identity_captured']}/{stats['n_identity_input']} "
              f"({stats['pct_identity_captured']:.1f}%)")
        
        print(f"\n★ Conversion candidates (core + bridging): {stats['n_conversion_candidates']}")
        candidates = self.results['conversion_candidates']
        if len(candidates) > 15:
            print(f"    {candidates[:15]} ... (+{len(candidates)-15} more)")
        else:
            print(f"    {candidates}")
        
        print(f"\nNetwork: {stats['network_nodes']} nodes, {stats['network_edges']} edges")
        print(f"Communities found: {stats['n_communities']}")
        
        print("\n" + "="*70)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_infomap_prototype(
    identity_tfs: List[str],
    high_exp_tfs: List[str],
    chipseq_file: str,
    adata = None,
    target_cell_type: str = None,
    cell_type_key: str = 'cell_type',
    corr_threshold: float = 0.3,
    pkn: pd.DataFrame = None,
    min_community_size: int = 3,
    verbose: bool = True
):
    """
    Test the InfoMap prototype.
    
    Uses identity-anchored PKN edges + high_exp filter + correlation filter.
    
    Usage:
        results = test_infomap_prototype(
            identity_tfs=stage2_results['identity_tfs'],
            high_exp_tfs=stage2_results['high_exp_tfs'],
            chipseq_file='/path/to/chipseq.tsv',
            adata=merged_adata,
            target_cell_type='dopaminergic neuron',
            corr_threshold=0.3
        )
    """
    detector = StageIIICoreIdentifier(
        identity_tfs=identity_tfs,
        high_exp_tfs=high_exp_tfs,
        chipseq_file=chipseq_file,
        pkn=pkn,
        adata=adata,
        target_cell_type=target_cell_type,
        cell_type_key=cell_type_key,
        corr_threshold=corr_threshold,
        min_community_size=min_community_size,
        verbose=verbose
    )
    
    return detector.run()


# =============================================================================
# BACKWARD COMPATIBLE API (matches grn_base.py)
# =============================================================================

def run_stage_iii(
    tf_results: Dict,
    chipseq_file: str = None,
    pkn: pd.DataFrame = None,
    adata = None,
    target_cell_type: str = None,
    cell_type_key: str = 'cell_type',
    corr_threshold: float = 0.3,
    corr_method: str = 'spearman',
    min_community_size: int = 3,
    verbose: bool = True,
    # Ignored parameters (for compatibility with grn_base.py)
    use_correlation_filter: bool = True,
    min_scc_size: int = 2,
    max_hops: int = 2,
    **kwargs
) -> Dict:
    """
    Run Stage III using InfoMap community detection.
    
    Backward compatible with grn_base.py API.
    
    Parameters
    ----------
    tf_results : dict
        Results from Stage I/II containing:
        - 'identity_tfs': List of identity TFs
        - 'high_exp_tfs': List of highly expressed TFs
    chipseq_file : str
        Path to ChIP-seq PKN file
    pkn : pd.DataFrame, optional
        Pre-loaded PKN dataframe
    adata : AnnData, optional
        Expression data for correlation filtering
    target_cell_type : str, optional
        Target cell type for correlation computation
    cell_type_key : str
        Column in adata.obs for cell type
    corr_threshold : float
        Minimum correlation to keep edge (default 0.3)
    corr_method : str
        'spearman' or 'pearson'
    min_community_size : int
        Minimum community size for InfoMap
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results dictionary compatible with grn_base.py output
    """
    # Extract from tf_results
    identity_tfs = tf_results.get('identity_tfs', [])
    high_exp_tfs = tf_results.get('high_exp_tfs', [])
    
    if not identity_tfs:
        raise ValueError("tf_results must contain 'identity_tfs'")
    if not high_exp_tfs:
        raise ValueError("tf_results must contain 'high_exp_tfs'")
    
    # Run InfoMap
    detector = StageIIICoreIdentifier(
        identity_tfs=identity_tfs,
        high_exp_tfs=high_exp_tfs,
        chipseq_file=chipseq_file,
        pkn=pkn,
        adata=adata,
        target_cell_type=target_cell_type,
        cell_type_key=cell_type_key,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        min_community_size=min_community_size,
        verbose=verbose
    )
    
    return detector.run()


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """
    Convert results to a DataFrame for easy inspection.
    Compatible with grn_base.py format.
    """
    # Collect all identity TFs with their classification
    all_identity = set(results['identity_tfs_input'])
    
    rows = []
    for tf in sorted(all_identity):
        if tf in results['core_scc']:
            category = 'core'
        elif tf in results['downstream_tfs']:
            category = 'other_community'
        elif tf in results['isolated_tfs']:
            category = 'isolated'
        else:
            category = 'unknown'
        
        rows.append({
            'TF': tf,
            'category': category,
            'in_conversion_candidates': tf in results['conversion_candidates']
        })
    
    return pd.DataFrame(rows)

