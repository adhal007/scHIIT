from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd

class CXGSampler:
    def __init__(self, adata, embed_key, broad_group_key, sub_group_key, target_per_broad=10000):
        self.adata = adata
        self.target_per_broad = target_per_broad
        self.embed_key = embed_key
        self.broad_group_key = broad_group_key
        self.sub_group_key = sub_group_key
        self.broad_groups = list(self.adata.obs[self.broad_group_key].unique())


    def fast_representative_sampling(self, obs_subset, embedding_subset, n_target, group_name):
        n_available = len(obs_subset)
        print(f"  {group_name}: {n_available} → {n_target}")
        
        if n_available <= n_target:
            print(f"    → Keeping all")
            return obs_subset.index.tolist()
        
        n_clusters = min(n_target, n_available // 10)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=1000,
            n_init=3
        )
        
        clusters = kmeans.fit_predict(embedding_subset)
        
        sampled_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            if cluster_mask.sum() == 0:
                continue
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embedding = embedding_subset[cluster_mask]
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embedding - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            sampled_indices.append(closest_idx)
        
        return obs_subset.index[sampled_indices].tolist()

    def run(self, return_full=False):      
        # Embedding
        if self.adata.obsm:
            available_embed_keys = list(self.adata.obsm.keys())
            embedding_key = self.embed_key if self.embed_key in available_embed_keys else available_embed_keys[0]

        # # Sampling
        # TARGET_PER_CLASS = 10000
        np.random.seed(52)

        all_sampled_indices = []

        for class_name in self.broad_groups:
            class_mask = self.adata.obs[self.broad_group_key] == class_name
            
            print(f"\n{'='*60}")
            print(f"{class_name}: {class_mask.sum()} cells → {self.target_per_broad}")
            print(f"{'='*60}")
            
            subclasses = self.adata.obs[class_mask][self.sub_group_key].unique()
            per_subclass = self.target_per_broad// len(subclasses)
            
            for subclass in subclasses:
                subclass_mask = (self.adata.obs[self.broad_group_key] == class_name) & (self.adata.obs[self.sub_group_key] == subclass) & (self.adata.obs['disease'] == 'normal')
                
                obs_subset = self.adata.obs[subclass_mask]
                embedding_subset = self.adata.obsm[self.embed_key][subclass_mask]
                
                sampled = self.fast_representative_sampling(obs_subset, embedding_subset, per_subclass, subclass)
                all_sampled_indices.extend(sampled)
        if return_full:
            print(f"\nTOTAL: {len(all_sampled_indices)} cells")
            self.sampled_adata = self.adata[all_sampled_indices].to_memory()
            return self.sampled_adata
        
        else:
            return all_sampled_indices