import numpy as np
import pandas as pd
from typing import List, Tuple

class ParetoGeneSelector:
    """
    Multi-objective gene selection using NSGA-II algorithm.
    
    Implements:
    - Non-dominated sorting (Pareto fronts)
    - Three ranking methods: crowding distance, distance to ideal, hypervolume
    - Relaxed Pareto selection across multiple fronts
    """
    
    def __init__(self, min_direction=0.0, min_mean_target=0.1):
        """
        Args:
            min_direction: Minimum direction threshold (>0 = upregulated in target)
            min_mean_target: Minimum mean expression in target cells
        """
        self.min_direction = min_direction
        self.min_mean_target = min_mean_target
    
    def select_top_genes(
        self,
        scores_df: pd.DataFrame,
        top_n: int = 20,
        ranking_method: str = 'crowding_distance'
    ) -> List[str]:
        """
        NSGA-II: Select top N genes using non-dominated sorting + ranking.
        
        Algorithm:
        1. Pre-filter by direction and mean expression
        2. Sort genes into Pareto fronts (Front 1, 2, 3, ...)
        3. Fill selection with Front 1, then Front 2, etc.
        4. Within last front, rank by chosen method
        
        Args:
            scores_df: DataFrame with columns:
                - gene (or as index)
                - gjsd_score
                - signed_specificity
                - specificity_target
                - mean_target
                - direction
            top_n: Number of genes to select
            ranking_method: 'crowding_distance', 'distance_to_ideal', or 'hypervolume'
        
        Returns:
            List of selected gene names
        """
        # Ensure 'gene' is a column, not index
        if 'gene' not in scores_df.columns:
            scores_df = scores_df.reset_index()
        
        # Pre-filter
        filtered = scores_df[
            (scores_df['direction'] > self.min_direction) &
            (scores_df['mean_target'] > self.min_mean_target)
        ].copy()
        
        if len(filtered) == 0:
            print("  ⚠️ No genes pass pre-filter!")
            return []
        
        print(f"  Pre-filter: {len(filtered)} genes (direction > {self.min_direction}, mean > {self.min_mean_target})")
        
        # Extract objectives (all to maximize)
        objectives = filtered[[
            'gjsd_score',
            'signed_specificity', 
            'specificity_target',
            'mean_target'
        ]].values
        
        # Step 1: Non-dominated sorting (assign fronts)
        fronts = self._fast_non_dominated_sort(objectives)
        
        print(f"  Non-dominated sorting: {len(fronts)} fronts")
        for i, front in enumerate(fronts[:5], 1):
            print(f"    Front {i}: {len(front)} genes")
            if i >= 5 and len(fronts) > 5:
                print(f"    ... ({len(fronts) - 5} more fronts)")
                break
        
        # Step 2: Fill selection from fronts
        selected_indices = []
        
        for front_idx, front in enumerate(fronts):
            if len(selected_indices) + len(front) <= top_n:
                # Add entire front
                selected_indices.extend(front)
                print(f"    Front {front_idx + 1}: Added all {len(front)} genes")
            else:
                # Last front: rank by chosen method
                remaining = top_n - len(selected_indices)
                front_objectives = objectives[front]
                
                # Compute ranking scores
                if ranking_method == 'crowding_distance':
                    scores = self._crowding_distance(front_objectives)
                elif ranking_method == 'distance_to_ideal':
                    scores = self._distance_to_ideal(front_objectives)
                elif ranking_method == 'hypervolume':
                    scores = self._hypervolume_contribution(front_objectives)
                else:
                    raise ValueError(f"Unknown ranking method: {ranking_method}")
                
                # Sort front by scores (descending)
                sorted_front = [front[i] for i in np.argsort(-scores)]
                selected_indices.extend(sorted_front[:remaining])
                
                print(f"    Front {front_idx + 1}: Selected {remaining}/{len(front)} genes by {ranking_method}")
                break
            
            if len(selected_indices) >= top_n:
                break
        
        # Convert indices to gene names
        selected_genes = filtered.iloc[selected_indices]['gene'].tolist()
        
        print(f"  ✓ Selected {len(selected_genes)} genes total")
        
        return selected_genes
    
    # ========================================================================
    # NON-DOMINATED SORTING (NSGA-II)
    # ========================================================================
    
    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """
        Fast non-dominated sorting algorithm (NSGA-II).
        
        Complexity: O(MN^2) where M = objectives, N = solutions
        
        Args:
            objectives: (n_solutions, n_objectives) array to maximize
        
        Returns:
            List of fronts, where each front is a list of solution indices
        """
        n = len(objectives)
        
        # For each solution, count how many solutions dominate it
        domination_count = np.zeros(n, dtype=int)
        
        # For each solution, store which solutions it dominates
        dominated_solutions = [[] for _ in range(n)]
        
        # Compute domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                # Does i dominate j?
                # i dominates j if: i >= j on all objectives AND i > j on at least one
                i_dominates_j = (
                    np.all(objectives[i] >= objectives[j]) and
                    np.any(objectives[i] > objectives[j])
                )
                
                # Does j dominate i?
                j_dominates_i = (
                    np.all(objectives[j] >= objectives[i]) and
                    np.any(objectives[j] > objectives[i])
                )
                
                if i_dominates_j:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif j_dominates_i:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # Build fronts iteratively
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            
            # For each solution in current front, reduce domination count of dominated solutions
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        return fronts
    
    # ========================================================================
    # RANKING METHOD 1: CROWDING DISTANCE (NSGA-II)
    # ========================================================================
    
    def _crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        """
        Compute crowding distance for each solution (NSGA-II).
        
        Measures how isolated a solution is in objective space.
        Higher = more isolated = better diversity.
        
        Args:
            objectives: (n_solutions, n_objectives) array
        
        Returns:
            Array of crowding distances (higher = better)
        """
        n, m = objectives.shape
        distances = np.zeros(n)
        
        for obj_idx in range(m):
            # Sort by this objective
            sorted_indices = np.argsort(objectives[:, obj_idx])
            
            # Boundary solutions get infinite distance (always selected)
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Normalize by range
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            
            if obj_range == 0:
                continue
            
            # Compute crowding distance for interior solutions
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    objectives[sorted_indices[i + 1], obj_idx] - 
                    objectives[sorted_indices[i - 1], obj_idx]
                ) / obj_range
        
        return distances
    
    # ========================================================================
    # RANKING METHOD 2: DISTANCE TO IDEAL POINT
    # ========================================================================
    
    def _distance_to_ideal(self, objectives: np.ndarray) -> np.ndarray:
        """
        Rank by Euclidean distance to ideal point (utopia point).
        
        Ideal point = best value on each objective in this front.
        Favors solutions closest to "perfect" gene.
        
        Args:
            objectives: (n_solutions, n_objectives) array
        
        Returns:
            Array of scores (higher = closer to ideal)
        """
        # Normalize objectives to [0, 1]
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        
        # Avoid division by zero
        obj_range[obj_range == 0] = 1.0
        obj_norm = (objectives - obj_min) / obj_range
        
        # Ideal point (1 on all objectives)
        ideal = np.ones(obj_norm.shape[1])
        
        # Euclidean distance to ideal
        distances = np.linalg.norm(obj_norm - ideal, axis=1)
        
        # Convert to scores (smaller distance = higher score)
        scores = 1.0 / (distances + 1e-10)
        
        return scores
    
    # ========================================================================
    # RANKING METHOD 3: HYPERVOLUME CONTRIBUTION
    # ========================================================================
    
    def _hypervolume_contribution(self, objectives: np.ndarray) -> np.ndarray:
        """
        Compute hypervolume contribution for each solution.
        
        Hypervolume = volume of objective space dominated by solution.
        Contribution = loss in hypervolume if solution is removed.
        
        Uses exact algorithm for 2D, Monte Carlo for 3D+.
        
        Args:
            objectives: (n_solutions, n_objectives) array
        
        Returns:
            Array of hypervolume contributions (higher = better)
        """
        n_solutions, n_objectives = objectives.shape
        
        # Normalize objectives to [0, 1]
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0
        obj_norm = (objectives - obj_min) / obj_range
        
        # Reference point (nadir - slightly below minimum)
        ref_point = np.zeros(n_objectives) - 0.1
        
        if n_objectives == 2:
            # Exact 2D hypervolume (fast)
            return self._hypervolume_2d_exact(obj_norm, ref_point)
        else:
            # Monte Carlo approximation for 3D+ (slower but feasible)
            return self._hypervolume_monte_carlo(obj_norm, ref_point)
    
    def _hypervolume_2d_exact(self, objectives: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
        """
        Exact 2D hypervolume contribution using sweep-line algorithm.
        
        Complexity: O(N log N)
        """
        n_solutions = len(objectives)
        
        # Sort by first objective (descending)
        sorted_idx = np.argsort(-objectives[:, 0])
        sorted_obj = objectives[sorted_idx]
        
        # Compute total hypervolume
        total_hv = 0.0
        for i in range(n_solutions):
            if i == 0:
                width = sorted_obj[i, 0] - ref_point[0]
            else:
                width = sorted_obj[i, 0] - sorted_obj[i-1, 0]
            
            height = sorted_obj[i, 1] - ref_point[1]
            total_hv += max(0, width * height)
        
        # Compute contribution of each point
        contributions = np.zeros(n_solutions)
        
        for i in range(n_solutions):
            # Hypervolume without point i
            hv_without = 0.0
            prev_x = ref_point[0]
            
            for j in range(n_solutions):
                if sorted_idx[j] == i:
                    continue  # Skip point i
                
                width = sorted_obj[j, 0] - prev_x
                height = sorted_obj[j, 1] - ref_point[1]
                hv_without += max(0, width * height)
                prev_x = sorted_obj[j, 0]
            
            # Contribution = total - without
            contributions[i] = total_hv - hv_without
        
        # Map back to original order
        result = np.zeros(n_solutions)
        for i, idx in enumerate(sorted_idx):
            result[idx] = contributions[i]
        
        return result
    
    def _hypervolume_monte_carlo(
        self, 
        objectives: np.ndarray, 
        ref_point: np.ndarray,
        n_samples: int = 10000
    ) -> np.ndarray:
        """
        Monte Carlo approximation of hypervolume contribution.
        
        For 3D+ objectives where exact computation is O(N^(M-1)).
        
        Algorithm:
        1. Sample random points in objective space
        2. Check which solutions dominate each sample
        3. Contribution = fraction of space uniquely dominated by solution
        
        Complexity: O(N * n_samples * M)
        """
        n_solutions, n_objectives = objectives.shape
        
        # Find bounds
        obj_max = objectives.max(axis=0)
        
        # Sample random points uniformly in [ref_point, obj_max]
        samples = np.random.uniform(
            low=ref_point,
            high=obj_max,
            size=(n_samples, n_objectives)
        )
        
        # For each sample, check which solutions dominate it
        dominated_by = np.zeros((n_samples, n_solutions), dtype=bool)
        
        for i in range(n_solutions):
            # Solution i dominates sample if objectives[i] >= sample on all dimensions
            dominated_by[:, i] = np.all(objectives[i] >= samples, axis=1)
        
        # Contribution of each solution
        contributions = np.zeros(n_solutions)
        
        for i in range(n_solutions):
            # Samples dominated only by solution i (not by any other)
            mask = np.ones(n_solutions, dtype=bool)
            mask[i] = False
            
            dominated_by_others = dominated_by[:, mask].any(axis=1)
            dominated_only_by_i = dominated_by[:, i] & (~dominated_by_others)
            
            # Contribution = fraction of space uniquely covered
            contributions[i] = dominated_only_by_i.sum() / n_samples
        
        return contributions
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def compare_ranking_methods(
        self,
        scores_df: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Compare all three ranking methods.
        
        Returns:
            DataFrame with gene rankings from each method
        """
        results = {}
        
        for method in ['crowding_distance', 'distance_to_ideal', 'hypervolume']:
            print(f"\n{'='*80}")
            print(f"METHOD: {method}")
            print('='*80)
            
            genes = self.select_top_genes(
                scores_df=scores_df,
                top_n=top_n,
                ranking_method=method
            )
            
            results[method] = genes
        
        # Create comparison DataFrame
        max_len = max(len(g) for g in results.values())
        comparison = pd.DataFrame({
            method: genes + [''] * (max_len - len(genes))
            for method, genes in results.items()
        })
        
        # Compute overlaps
        print(f"\n{'='*80}")
        print("OVERLAP ANALYSIS")
        print('='*80)
        
        from itertools import combinations
        for m1, m2 in combinations(results.keys(), 2):
            overlap = len(set(results[m1]) & set(results[m2]))
            print(f"{m1:20s} vs {m2:20s}: {overlap}/{top_n} overlap")
        
        return comparison