"""
Simple Core TF Network Visualization
=====================================

Minimalist code to visualize the core TF network from Stage III results.
Just 4 simple functions - no classes.
"""

import matplotlib.pyplot as plt
import networkx as nx

"""
Simple Core TF Network Visualization
=====================================

Minimalist code to visualize the core TF network from Stage III results.
Just 4 simple functions - no classes.
"""

import matplotlib.pyplot as plt
import networkx as nx

def plot_core_network(stage3_results, save_path='core_network.png'):
    """
    Plot the core TF network.
    
    Args:
        stage3_results: Output from run_stage_iii()
        save_path: Where to save the plot
    """
    # Extract data
    G = stage3_results['network']
    core_tfs = stage3_results['core_tfs']
    unique_tfs = set(stage3_results['unique_tfs'])
    
    # Get subgraph of core TFs only
    G_core = G.subgraph(core_tfs).copy()
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G_core, k=0.5, iterations=50, seed=42)
    
    # Separate nodes by type
    unique_nodes = [n for n in G_core.nodes() if n in unique_tfs]
    connecting_nodes = [n for n in G_core.nodes() if n not in unique_tfs]
    
    # Draw edges
    nx.draw_networkx_edges(G_core, pos, alpha=0.3, arrows=True, 
                          arrowsize=15, width=1.5, 
                          edge_color='gray', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_core, pos, nodelist=unique_nodes,
                          node_color='#E74C3C', node_size=500,
                          label='Unique TFs', ax=ax)
    
    nx.draw_networkx_nodes(G_core, pos, nodelist=connecting_nodes,
                          node_color='#3498DB', node_size=500,
                          label='Connecting TFs', ax=ax)
    
    # Labels - always show them
    nx.draw_networkx_labels(G_core, pos, font_size=8, 
                           font_weight='bold', ax=ax)
    
    # Title and legend
    ax.set_title(f'Core TF Regulatory Network\n'
                f'{len(G_core)} TFs, {G_core.number_of_edges()} interactions\n'
                f'{len(unique_nodes)} unique + {len(connecting_nodes)} connecting',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Network saved to: {save_path}")
    
    return fig


def plot_scc_comparison(stage3_results, save_path='scc_comparison.png'):
    """
    Plot comparison of all SCCs side by side.
    
    Args:
        stage3_results: Output from run_stage_iii()
        save_path: Where to save the plot
    """
    G = stage3_results['network']
    sccs = stage3_results['sccs'][:4]  # Top 4 SCCs
    unique_tfs = set(stage3_results['unique_tfs'])
    
    n_sccs = len(sccs)
    fig, axes = plt.subplots(1, n_sccs, figsize=(5*n_sccs, 5))
    if n_sccs == 1:
        axes = [axes]
    
    for i, scc in enumerate(sccs):
        ax = axes[i]
        G_scc = G.subgraph(scc).copy()
        
        # Layout
        pos = nx.spring_layout(G_scc, k=0.8, iterations=50)
        
        # Node colors
        unique_nodes = [n for n in G_scc.nodes() if n in unique_tfs]
        connecting_nodes = [n for n in G_scc.nodes() if n not in unique_tfs]
        
        # Draw
        nx.draw_networkx_edges(G_scc, pos, alpha=0.4, arrows=True, ax=ax)
        nx.draw_networkx_nodes(G_scc, pos, nodelist=unique_nodes,
                              node_color='#E74C3C', node_size=300, ax=ax)
        nx.draw_networkx_nodes(G_scc, pos, nodelist=connecting_nodes,
                              node_color='#3498DB', node_size=300, ax=ax)
        
        if len(G_scc) <= 15:
            nx.draw_networkx_labels(G_scc, pos, font_size=6, ax=ax)
        
        ax.set_title(f'SCC {i+1}\n{len(G_scc)} TFs', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ SCC comparison saved to: {save_path}")
    
    return fig


def plot_network_stats(stage3_results, save_path='network_stats.png'):
    """
    Plot network statistics and degree distributions.
    
    Args:
        stage3_results: Output from run_stage_iii()
        save_path: Where to save the plot
    """
    G = stage3_results['network']
    core_tfs = stage3_results['core_tfs']
    G_core = G.subgraph(core_tfs)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. SCC size distribution
    ax = axes[0, 0]
    scc_sizes = [len(scc) for scc in stage3_results['sccs']]
    ax.bar(range(len(scc_sizes)), scc_sizes, color='#3498DB')
    ax.set_xlabel('SCC Rank', fontsize=11)
    ax.set_ylabel('Size', fontsize=11)
    ax.set_title('SCC Size Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 2. Degree distribution
    ax = axes[0, 1]
    in_degrees = [d for n, d in G_core.in_degree()]
    out_degrees = [d for n, d in G_core.out_degree()]
    ax.hist(in_degrees, bins=20, alpha=0.6, label='In-degree', color='#E74C3C')
    ax.hist(out_degrees, bins=20, alpha=0.6, label='Out-degree', color='#3498DB')
    ax.set_xlabel('Degree', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Top hub TFs
    ax = axes[1, 0]
    total_degrees = [(n, G_core.in_degree(n) + G_core.out_degree(n)) 
                     for n in G_core.nodes()]
    top_hubs = sorted(total_degrees, key=lambda x: x[1], reverse=True)[:15]
    nodes, degrees = zip(*top_hubs)
    ax.barh(range(len(nodes)), degrees, color='#2ECC71')
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=8)
    ax.set_xlabel('Total Degree', fontsize=11)
    ax.set_title('Top 15 Hub TFs', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    # 4. Network summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    CORE NETWORK STATISTICS
    
    Nodes: {G_core.number_of_nodes()}
    Edges: {G_core.number_of_edges()}
    Density: {nx.density(G_core):.4f}
    
    Avg in-degree: {sum(d for n, d in G_core.in_degree()) / G_core.number_of_nodes():.2f}
    Avg out-degree: {sum(d for n, d in G_core.out_degree()) / G_core.number_of_nodes():.2f}
    
    # SCCs: {len(stage3_results['sccs'])}
    Largest SCC: {len(stage3_results['largest_scc'])}
    
    Unique TFs: {len(set(stage3_results['unique_tfs']) & set(core_tfs))}
    Connecting TFs: {len(set(stage3_results['connecting_tfs']) & set(core_tfs))}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Network stats saved to: {save_path}")
    
    return fig


def plot_circular_network(stage3_results, save_path='core_network_circular.png'):
    """
    Plot core network in circular layout - good for seeing feedback loops.
    
    Args:
        stage3_results: Output from run_stage_iii()
        save_path: Where to save the plot
    """
    G = stage3_results['network']
    core_tfs = stage3_results['core_tfs']
    unique_tfs = set(stage3_results['unique_tfs'])
    
    G_core = G.subgraph(core_tfs).copy()
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Circular layout
    pos = nx.circular_layout(G_core)
    
    # Separate by type
    unique_nodes = [n for n in G_core.nodes() if n in unique_tfs]
    connecting_nodes = [n for n in G_core.nodes() if n not in unique_tfs]
    
    # Draw edges with curved style for feedback loops
    nx.draw_networkx_edges(G_core, pos, alpha=0.2, arrows=True,
                          arrowsize=12, width=1.0,
                          edge_color='gray', 
                          connectionstyle='arc3,rad=0.1',
                          ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_core, pos, nodelist=unique_nodes,
                          node_color='#E74C3C', node_size=600,
                          label='Unique TFs', ax=ax)
    
    nx.draw_networkx_nodes(G_core, pos, nodelist=connecting_nodes,
                          node_color='#3498DB', node_size=600,
                          label='Connecting TFs', ax=ax)
    
    # Labels
    nx.draw_networkx_labels(G_core, pos, font_size=7, 
                           font_weight='bold', ax=ax)
    
    ax.set_title(f'Core TF Regulatory Network (Circular Layout)\n'
                f'Feedback loops visible as curved edges',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Circular network saved to: {save_path}")
    
    return fig



