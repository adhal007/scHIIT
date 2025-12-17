from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Optional
from pronto import Ontology
import pandas as pd
from pathlib import Path
import json
from typing import Optional, Dict

from pronto import Ontology
import pandas as pd
from pathlib import Path
import json
from typing import Optional, Dict

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Optional, Set
from pronto import Ontology
import pandas as pd
from pathlib import Path
import json
from typing import Optional, Dict

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import json


class SapBERTOptimizedDescriptionGenerator:
    """
    Generate cell type descriptions optimized for SapBERT
    Focus on: entities, relationships, markers, ontology terms
    """
    def __init__(self, metadata_dir: str):
        self.metadata_dir = Path(metadata_dir)
        
        # Load Cell Ontology
        print("Loading Cell Ontology from OBO Library...")
        try:
            self.cell_onto = Ontology.from_obo_library("cl.owl")
            print(f"✓ Loaded Cell Ontology: {len(self.cell_onto)} terms")
        except Exception as e:
            print(f"⚠ Warning: Could not load Cell Ontology: {e}")
            self.cell_onto = None
    
    def get_ontology_info(self, ontology_id: str) -> Optional[Dict]:
        """Extract ontology information"""
        if not self.cell_onto or ontology_id == 'unknown':
            return None
        
        try:
            term = self.cell_onto[ontology_id]
            
            definition = str(term.definition) if term.definition else None
            
            parents = []
            for parent in term.superclasses(distance=1, with_self=False):
                if parent.id.startswith('CL:'):
                    parents.append({
                        'id': parent.id,
                        'name': parent.name
                    })
            
            synonyms = [str(syn) for syn in term.synonyms] if term.synonyms else []
            
            relationships = []
            for rel in term.relationships:
                rel_type = str(rel)
                for target in term.relationships[rel]:
                    if target.id.startswith('CL:'):
                        relationships.append({
                            'type': rel_type,
                            'target': target.name
                        })
            
            return {
                'name': term.name,
                'definition': definition,
                'parents': parents,
                'synonyms': synonyms,
                'relationships': relationships
            }
            
        except (KeyError, Exception):
            return None
    
    def create_sapbert_description(self, 
                                   cell_type_name: str,
                                   cell_type_info: dict,
                                   dataset_metadata: dict) -> str:
        """
        Create SapBERT-optimized description
        
        SapBERT works best with:
        - Structured entity descriptions
        - Clear biological terminology
        - Marker genes/proteins
        - Functional characteristics
        - Ontology relationships
        """
        ontology_id = cell_type_info['ontology_id']
        n_cells = cell_type_info['n_cells']
        
        # Dataset context
        organ = dataset_metadata['biological']['organ']
        tissue = dataset_metadata['biological']['tissue']
        organism = dataset_metadata['biological']['organism']
        disease = dataset_metadata['biological']['disease']
        
        # Get ontology info
        onto_info = self.get_ontology_info(ontology_id)
        
        # Build structured description
        sections = []
        
        # 1. ENTITY HEADER (most important for SapBERT)
        sections.append(f"Cell type: {cell_type_name}")
        sections.append(f"Ontology: {ontology_id}")
        
        # 2. SYNONYMS (SapBERT is trained on synonyms)
        if onto_info and onto_info['synonyms']:
            syn_list = ', '.join(onto_info['synonyms'][:4])
            sections.append(f"Synonyms: {syn_list}")
        
        # 3. DEFINITION (from Cell Ontology - authoritative)
        if onto_info and onto_info['definition']:
            # Keep definition concise (SapBERT works better with shorter texts)
            definition = onto_info['definition']
            if len(definition) > 250:
                definition = definition[:247] + "..."
            sections.append(f"Definition: {definition}")
        
        # 4. LINEAGE (parent cell types - important for hierarchy)
        if onto_info and onto_info['parents']:
            parent_names = [p['name'] for p in onto_info['parents'][:3]]
            sections.append(f"Cell lineage: {' > '.join(parent_names)}")
        
        # 5. RELATIONSHIPS (develops_from, capable_of, etc.)
        if onto_info and onto_info['relationships']:
            develops_from = [r['target'] for r in onto_info['relationships'] 
                           if 'develops_from' in r['type']]
            if develops_from:
                sections.append(f"Develops from: {', '.join(develops_from[:2])}")
            
            capable_of = [r['target'] for r in onto_info['relationships'] 
                         if 'capable_of' in r['type']]
            if capable_of:
                sections.append(f"Functions: {', '.join(capable_of[:3])}")
        
        # 6. LOCATION/TISSUE CONTEXT
        if not tissue.startswith('mixed'):
            sections.append(f"Tissue location: {tissue}, {organ}")
        else:
            sections.append(f"Organ system: {organ}")
        
        # 7. ORGANISM
        sections.append(f"Species: {organism}")
        
        # 8. SAMPLE CHARACTERISTICS
        sections.append(f"Sample: {disease} tissue, {n_cells} cells")
        
        # Join with newlines (structured format works well with SapBERT)
        return "\n".join(sections)
    
    def generate_all_descriptions(self, save_cache: bool = True) -> pd.DataFrame:
        """Generate descriptions with proper file filtering"""
        all_descriptions = []
        ontology_cache = {}
        
        # FIXED: Only load organ_NNN.json files
        import re
        all_files = list(self.metadata_dir.glob("*.json"))
        pattern = re.compile(r'^(brain|blood|lung|kidney|pancreas|heart|liver|intestine|skin|reproductive)_\d+\.json$')
        dataset_files = [f for f in all_files if pattern.match(f.name)]
        
        print(f"\nGenerating SapBERT-optimized descriptions...")
        print(f"Found {len(dataset_files)} valid dataset files\n")
        
        for dataset_file in sorted(dataset_files):
            with open(dataset_file, 'r') as f:
                dataset_meta = json.load(f)
            
            dataset_id = dataset_meta['dataset_id']
            source_id = dataset_meta.get('source_id', 'unknown')
            organ = dataset_meta['biological']['organ']
            
            print(f"  {dataset_id}: {len(dataset_meta['cell_types'])} cell types")
            
            for ct_name, ct_info in dataset_meta['cell_types'].items():
                ontology_id = ct_info['ontology_id']
                clean_ct_name = ct_name.replace(' ', '_').replace(',', '').replace('-', '_').replace('/', '_')
                cell_type_id = f"{dataset_id}__{clean_ct_name}"
                
                if ontology_id not in ontology_cache:
                    onto_info = self.get_ontology_info(ontology_id)
                    ontology_cache[ontology_id] = onto_info
                else:
                    onto_info = ontology_cache[ontology_id]
                
                description = self.create_sapbert_description(ct_name, ct_info, dataset_meta)
                
                row = {
                    'cell_type_id': cell_type_id,
                    'cell_type_name': ct_name,
                    'dataset_id': dataset_id,
                    'source_id': source_id,
                    'organ': organ,
                    'tissue': dataset_meta['biological']['tissue'],
                    'ontology_id': ontology_id,
                    'n_cells': ct_info['n_cells'],
                    'description': description,
                    'description_type': 'sapbert_optimized'
                }
                
                if onto_info:
                    row['onto_definition'] = onto_info['definition']
                    row['onto_parents'] = json.dumps([p['name'] for p in onto_info['parents']])
                    row['onto_synonyms'] = json.dumps(onto_info['synonyms'])
                else:
                    row['onto_definition'] = None
                    row['onto_parents'] = None
                    row['onto_synonyms'] = None
                
                all_descriptions.append(row)
        
        df = pd.DataFrame(all_descriptions)
        
        print(f"\n{'='*100}")
        print(f"✓ Generated {len(df)} SapBERT-optimized descriptions")
        print(f"  - {df['onto_definition'].notna().sum()} with ontology definitions")
        print(f"{'='*100}")
        
        if save_cache:
            cache_path = self.metadata_dir / 'ontology_cache_sapbert.json'
            cache_serializable = {k: v for k, v in ontology_cache.items() if v is not None}
            with open(cache_path, 'w') as f:
                json.dump(cache_serializable, f, indent=2)
            print(f"  ✓ Saved cache: {cache_path}")
        
        return df
    
    def show_description_examples(self, descriptions_df: pd.DataFrame, n: int = 3):
        """
        Show example descriptions for review
        """
        print(f"\n{'='*100}")
        print(f"EXAMPLE SAPBERT-OPTIMIZED DESCRIPTIONS")
        print(f"{'='*100}\n")
        
        for i in range(min(n, len(descriptions_df))):
            row = descriptions_df.iloc[i]
            print(f"Example {i+1}:")
            print(f"Cell Type: {row['cell_type_name']}")
            print(f"Organ: {row['organ']}")
            print(f"\nDescription:")
            print("-" * 100)
            print(row['description'])
            print("-" * 100)
            print()

class SapBERTEmbeddingGenerator:
    """
    Generate embeddings using SapBERT with transformers library
    No sentence-transformers wrapper - direct implementation
    
    UPDATED: Now saves all files needed by FAISSIndexBuilder
    """
    def __init__(self, model_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext'):
        print(f"Loading SapBERT with transformers library...")
        print(f"  Model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = 768
        self.model_name = model_name
        
        print(f"  ✓ Model loaded on {self.device}")
        print(f"  Embedding dimension: {self.embedding_dim}")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings
        """
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings, weighted by attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum of attention mask (clamped to avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Mean pooling
        return sum_embeddings / sum_mask
    
    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress=True):
        """
        Encode texts to embeddings
        
        Parameters:
        -----------
        texts : list of str
            Texts to encode
        batch_size : int
            Batch size for encoding
        normalize_embeddings : bool
            Whether to L2 normalize embeddings
        show_progress : bool
            Whether to show progress bar
        
        Returns:
        --------
        np.ndarray : Embeddings [n_texts, embedding_dim]
        """
        all_embeddings = []
        
        # Create batches
        n_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Encoding")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Get embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Mean pooling
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            
            batch_embeddings = self.mean_pooling(token_embeddings, attention_mask)
            
            # Normalize
            if normalize_embeddings:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_descriptions(self, 
                          descriptions_df: pd.DataFrame,
                          batch_size: int = 32,
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate SapBERT embeddings for cell type descriptions
        """
        descriptions = descriptions_df['description'].tolist()
        
        print(f"\nGenerating SapBERT embeddings for {len(descriptions)} cell types...")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {self.device}")
        
        # Generate embeddings
        embeddings = self.encode(
            descriptions,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress=show_progress
        )
        
        print(f"\n  ✓ Embeddings generated")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")
        print(f"  Dtype: {embeddings.dtype}")
        
        # Verify normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Norm check: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        
        return embeddings
    
    def save_embeddings(self,
                       embeddings: np.ndarray,
                       descriptions_df: pd.DataFrame,
                       output_dir: str):
        """
        Save SapBERT embeddings and metadata
        
        UPDATED: Now saves all files needed by FAISSIndexBuilder:
        - embeddings.npy (for FAISSIndexBuilder compatibility)
        - embeddings_sapbert.npy (original name, kept for backwards compatibility)
        - cell_type_ids.txt (required by FAISSIndexBuilder)
        - cell_type_metadata.csv (required by FAISSIndexBuilder)
        - embedding_metadata_sapbert.json (original metadata file)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nSaving to {output_dir}/")
        
        # Save embeddings (both names for compatibility)
        embeddings_f32 = embeddings.astype('float32')
        
        # Primary name for FAISSIndexBuilder
        embeddings_path = output_dir / 'embeddings.npy'
        np.save(embeddings_path, embeddings_f32)
        print(f"  ✓ embeddings.npy ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
        
        # Also save with sapbert suffix for clarity
        embeddings_sapbert_path = output_dir / 'embeddings_sapbert.npy'
        np.save(embeddings_sapbert_path, embeddings_f32)
        print(f"  ✓ embeddings_sapbert.npy (copy)")
        
        # Save cell type IDs (required by FAISSIndexBuilder)
        cell_type_ids_path = output_dir / 'cell_type_ids.txt'
        with open(cell_type_ids_path, 'w') as f:
            for cell_type_id in descriptions_df['cell_type_id']:
                f.write(f"{cell_type_id}\n")
        print(f"  ✓ cell_type_ids.txt ({len(descriptions_df)} IDs)")
        
        # Save metadata CSV (required by FAISSIndexBuilder)
        metadata_csv_path = output_dir / 'cell_type_metadata.csv'
        descriptions_df.to_csv(metadata_csv_path, index=False)
        print(f"  ✓ cell_type_metadata.csv ({len(descriptions_df)} rows)")
        
        # Save embedding metadata JSON
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'n_cell_types': len(embeddings),
            'normalized': True,
            'created_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'model_type': 'SapBERT',
            'implementation': 'transformers_direct',
            'device': str(self.device),
            'statistics': {
                'total_cell_types': len(embeddings),
                'total_cells': int(descriptions_df['n_cells'].sum()),
                'unique_datasets': descriptions_df['dataset_id'].nunique(),
                'organs': descriptions_df['organ'].unique().tolist()
            }
        }
        
        metadata_path = output_dir / 'embedding_metadata_sapbert.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ embedding_metadata_sapbert.json")
        
        print(f"\n{'='*100}")
        print(f"SapBERT embeddings saved successfully!")
        print(f"Files created:")
        print(f"  - embeddings.npy (for FAISSIndexBuilder)")
        print(f"  - embeddings_sapbert.npy (backup)")
        print(f"  - cell_type_ids.txt (for FAISSIndexBuilder)")
        print(f"  - cell_type_metadata.csv (for FAISSIndexBuilder)")
        print(f"  - embedding_metadata_sapbert.json")
        print(f"{'='*100}")
        
        return {
            'embeddings_path': embeddings_path,
            'embeddings_sapbert_path': embeddings_sapbert_path,
            'cell_type_ids_path': cell_type_ids_path,
            'metadata_csv_path': metadata_csv_path,
            'metadata_json_path': metadata_path
        }
    
    def validate_embeddings(self, embeddings: np.ndarray) -> dict:
        """
        Validate generated embeddings
        """
        print("\nValidating embeddings...")
        
        results = {
            'shape_correct': embeddings.shape[1] == self.embedding_dim,
            'no_nans': not np.isnan(embeddings).any(),
            'no_infs': not np.isinf(embeddings).any(),
            'normalized': np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
        }
        
        # Check for duplicates
        unique_embeddings = np.unique(embeddings, axis=0)
        results['n_unique'] = len(unique_embeddings)
        results['has_duplicates'] = len(unique_embeddings) < len(embeddings)
        
        # Statistics
        norms = np.linalg.norm(embeddings, axis=1)
        results['mean_norm'] = float(norms.mean())
        results['std_norm'] = float(norms.std())
        
        # Print results
        print(f"  Shape correct: {results['shape_correct']}")
        print(f"  No NaNs: {results['no_nans']}")
        print(f"  No Infs: {results['no_infs']}")
        print(f"  Normalized: {results['normalized']}")
        print(f"  Unique embeddings: {results['n_unique']}/{len(embeddings)}")
        
        if results['has_duplicates']:
            n_dupes = len(embeddings) - results['n_unique']
            print(f"  ⚠ Warning: {n_dupes} duplicate embeddings detected")
        
        all_pass = all([
            results['shape_correct'],
            results['no_nans'],
            results['no_infs'],
            results['normalized']
        ])
        
        if all_pass:
            print(f"  ✓ All validation checks passed!")
        else:
            print(f"  ✗ Some validation checks failed!")
        
        return results

class FAISSIndexBuilder:
    """
    Build FAISS index for fast semantic search
    
    UPDATED: More flexible file loading with automatic detection
    """
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        
        print("Loading embeddings...")
        
        # Load embeddings - try multiple file names
        embeddings_path = self._find_embeddings_file()
        self.embeddings = np.load(embeddings_path)
        print(f"  ✓ Loaded embeddings from: {embeddings_path.name}")
        
        # Load cell type IDs - try multiple approaches
        self.cell_type_ids = self._load_cell_type_ids()
        print(f"  ✓ Loaded {len(self.cell_type_ids)} cell type IDs")
        
        # Load metadata
        self.metadata_df = self._load_metadata()
        print(f"  ✓ Loaded metadata: {len(self.metadata_df)} rows")
        
        print(f"\n✓ Loaded {len(self.embeddings)} embeddings")
        print(f"  Shape: {self.embeddings.shape}")
    
    def _find_embeddings_file(self) -> Path:
        """Find embeddings file with fallback options"""
        # Priority order for embeddings files
        candidates = [
            'embeddings.npy',
            'embeddings_sapbert.npy',
            'cell_type_embeddings.npy'
        ]
        
        for filename in candidates:
            path = self.embeddings_dir / filename
            if path.exists():
                return path
        
        # If none found, list available .npy files
        npy_files = list(self.embeddings_dir.glob('*.npy'))
        if npy_files:
            raise FileNotFoundError(
                f"Could not find embeddings file. Expected one of: {candidates}\n"
                f"Available .npy files: {[f.name for f in npy_files]}"
            )
        else:
            raise FileNotFoundError(
                f"No .npy files found in {self.embeddings_dir}\n"
                f"Please run SapBERTEmbeddingGenerator.save_embeddings() first."
            )
    
    def _load_cell_type_ids(self) -> list:
        """Load cell type IDs with fallback options"""
        # Option 1: cell_type_ids.txt
        txt_path = self.embeddings_dir / 'cell_type_ids.txt'
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                return [line.strip() for line in f]
        
        # Option 2: Extract from metadata CSV
        csv_path = self.embeddings_dir / 'cell_type_metadata.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'cell_type_id' in df.columns:
                return df['cell_type_id'].tolist()
        
        # Option 3: Extract from descriptions CSV
        desc_path = self.embeddings_dir.parent / 'cell_type_descriptions_sapbert.csv'
        if desc_path.exists():
            df = pd.read_csv(desc_path)
            if 'cell_type_id' in df.columns:
                return df['cell_type_id'].tolist()
        
        raise FileNotFoundError(
            f"Could not find cell type IDs. Expected:\n"
            f"  - {self.embeddings_dir / 'cell_type_ids.txt'}, or\n"
            f"  - {self.embeddings_dir / 'cell_type_metadata.csv'} with 'cell_type_id' column\n"
            f"Please ensure SapBERTEmbeddingGenerator.save_embeddings() was run with the updated version."
        )
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata with fallback options"""
        # Option 1: cell_type_metadata.csv in embeddings dir
        csv_path = self.embeddings_dir / 'cell_type_metadata.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        
        # Option 2: descriptions CSV in parent dir
        desc_path = self.embeddings_dir.parent / 'cell_type_descriptions_sapbert.csv'
        if desc_path.exists():
            return pd.read_csv(desc_path)
        
        raise FileNotFoundError(
            f"Could not find metadata file. Expected:\n"
            f"  - {self.embeddings_dir / 'cell_type_metadata.csv'}, or\n"
            f"  - {self.embeddings_dir.parent / 'cell_type_descriptions_sapbert.csv'}\n"
            f"Please ensure SapBERTEmbeddingGenerator.save_embeddings() was run with the updated version."
        )
    
    def build_index(self, index_type='FlatIP'):
        """
        Build FAISS index
        
        Parameters:
        -----------
        index_type : str
            'FlatIP' - Exact search with inner product (cosine similarity)
            'FlatL2' - Exact search with L2 distance
            'IVFFlat' - Approximate search (faster for large datasets)
        """
        embedding_dim = self.embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity (if not already)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-5):
            print("Normalizing embeddings...")
            self.embeddings = self.embeddings / norms
        
        print(f"Building {index_type} index...")
        
        if index_type == 'FlatIP':
            index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == 'FlatL2':
            index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == 'IVFFlat':
            # For approximate search (faster for large datasets)
            n_cells = min(100, len(self.embeddings) // 10)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_cells)
            index.train(self.embeddings.astype('float32'))
        else:
            raise ValueError(f"Unknown index type: {index_type}. Use 'FlatIP', 'FlatL2', or 'IVFFlat'")
        
        # Add vectors to index
        index.add(self.embeddings.astype('float32'))
        
        print(f"✓ Index built: {index.ntotal} vectors")
        
        return index
    
    def save_index(self, index, output_dir: str):
        """Save FAISS index and metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        index_path = output_dir / 'faiss_index.bin'
        faiss.write_index(index, str(index_path))
        
        # Save index metadata
        metadata = {
            'index_type': type(index).__name__,
            'n_vectors': index.ntotal,
            'embedding_dim': self.embeddings.shape[1],
            'created_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'statistics': {
                'total_cell_types': len(self.embeddings),
                'total_cells': int(self.metadata_df['n_cells'].sum()) if 'n_cells' in self.metadata_df.columns else None,
                'unique_datasets': int(self.metadata_df['dataset_id'].nunique()) if 'dataset_id' in self.metadata_df.columns else None,
                'organs': self.metadata_df['organ'].unique().tolist() if 'organ' in self.metadata_df.columns else None
            }
        }
        
        meta_path = output_dir / 'index_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also copy cell type IDs and metadata to index directory for convenience
        cell_type_ids_path = output_dir / 'cell_type_ids.txt'
        with open(cell_type_ids_path, 'w') as f:
            for cell_type_id in self.cell_type_ids:
                f.write(f"{cell_type_id}\n")
        
        metadata_csv_path = output_dir / 'cell_type_metadata.csv'
        self.metadata_df.to_csv(metadata_csv_path, index=False)
        
        print(f"\n✓ Saved:")
        print(f"  - {index_path}")
        print(f"  - {meta_path}")
        print(f"  - {cell_type_ids_path}")
        print(f"  - {metadata_csv_path}")
        
        return index_path

# Improved SemanticReferenceRetriever with hybrid search capabilities
# Replace the existing class in cell_ontology_utils.py
class SemanticReferenceRetriever:
    """
    Semantic search for reference cell types
    
    IMPROVED VERSION with:
    - Hybrid search (semantic + keyword)
    - Keyword boosting
    - Negative keyword filtering
    - Better query handling
    """
    def __init__(self, base_dir: str, use_sapbert: bool = True, cl_obo_path: str = None):
        self.base_dir = Path(base_dir)
        
        print("Loading semantic search system...")
        
        # Choose model
        if use_sapbert:
            model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
            embeddings_file = 'embeddings_sapbert.npy'
            print(f"  Using: SapBERT")
        else:
            model_name = 'pritamdeka/S-PubMedBert-MS-MARCO'
            embeddings_file = 'embeddings.npy'
            print(f"  Using: PubMedBERT")

        # NEW: Load ontology for background selection
        if cl_obo_path:
            print("  Loading Cell Ontology for background selection...")
            self.ontology = OntologyBackgroundSelector(cl_obo_path)
            print(f" Ontology loaded ({len(self.ontology.terms)} terms)")
        else:
            print("  No CL ontology path provided - ontology features disabled")
            self.ontology = None

        # Load model
        self.model = SentenceTransformer(model_name)
        print("  Model loaded")
        
        # Load embeddings
        embeddings_path = self.base_dir / 'embeddings' / embeddings_file
        if not embeddings_path.exists():
            # Fallback to embeddings.npy
            embeddings_path = self.base_dir / 'embeddings' / 'embeddings.npy'
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found. Please generate embeddings first.")
        
        self.embeddings = np.load(embeddings_path)
        
        # Normalize if needed
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-5):
            self.embeddings = self.embeddings / norms
        
        print(f"  Embeddings loaded ({self.embeddings.shape})")
        
        # Load metadata
        metadata_path = self.base_dir / 'embeddings' / 'cell_type_metadata.csv'
        self.metadata_df = pd.read_csv(metadata_path)
        print(f"  Metadata loaded ({len(self.metadata_df)} cell types)")
        
        # Build keyword index for fast lookup
        self._build_keyword_index()
        
        print("\nReady for semantic search!")
    
    def _build_keyword_index(self):
        """Build inverted index for keyword search"""
        self.keyword_index = {}
        
        for idx, row in self.metadata_df.iterrows():
            # Tokenize cell type name
            cell_name = row['cell_type_name'].lower()
            tokens = set(cell_name.replace('-', ' ').replace('_', ' ').split())
            
            # Also add from ontology definition if available
            if pd.notna(row.get('onto_definition')):
                def_tokens = row['onto_definition'].lower().split()[:20]  # First 20 words
                tokens.update(def_tokens)
            
            for token in tokens:
                if token not in self.keyword_index:
                    self.keyword_index[token] = set()
                self.keyword_index[token].add(idx)
        
        print(f"  Keyword index built ({len(self.keyword_index)} unique terms)")
    
    def _get_keyword_matches(self, keywords: List[str]) -> Set[int]:
        """Get indices that match any of the keywords"""
        matches = set()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in self.keyword_index:
                matches.update(self.keyword_index[kw_lower])
        return matches
    
    def _keyword_score(self, row, positive_keywords: List[str], negative_keywords: List[str] = None) -> float:
        """
        Compute keyword-based score for a row
        
        Returns:
        - Positive score for matching positive keywords
        - Negative score (or filter) for matching negative keywords
        """
        cell_name = row['cell_type_name'].lower()
        definition = str(row.get('onto_definition', '')).lower()
        text = f"{cell_name} {definition}"
        
        score = 0.0
        
        # Positive keywords
        for kw in positive_keywords:
            if kw.lower() in text:
                # Boost more if in cell name vs definition
                if kw.lower() in cell_name:
                    score += 0.3  # Strong boost for name match
                else:
                    score += 0.1  # Smaller boost for definition match
        
        # Negative keywords
        if negative_keywords:
            for kw in negative_keywords:
                if kw.lower() in text:
                    score -= 0.5  # Penalty for negative keywords
        
        return score
    
    def search_hybrid(self,
                      query_text: str,
                      k: int = 10,
                      positive_keywords: List[str] = None,
                      negative_keywords: List[str] = None,
                      keyword_weight: float = 0.3,
                      organ_filter: str = None,
                      min_cells: int = None,
                      min_specificity: int = 0,
                      exclude_broad_terms: bool = True,
                      strict_negative_filter: bool = True) -> pd.DataFrame:
        """
        Hybrid search combining semantic similarity with keyword matching
        
        Parameters:
        -----------
        query_text : str
            Query for semantic search
        positive_keywords : List[str]
            Keywords that SHOULD appear (boost score)
        negative_keywords : List[str]
            Keywords that should NOT appear (reduce score or filter out)
        keyword_weight : float
            Weight for keyword score (0-1). Higher = more keyword influence
        strict_negative_filter : bool
            If True, completely exclude results with negative keywords
            If False, just reduce their score
        
        Example:
        --------
        retriever.search_hybrid(
            "inhibitory neurons",
            positive_keywords=["GABA", "GABAergic", "inhibitory"],
            negative_keywords=["glutamatergic", "excitatory"],
            k=10
        )
        """
        # Embed query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute semantic similarities
        semantic_scores = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top candidates (more than k for post-filtering)
        search_k = min(k * 20, len(semantic_scores))
        top_indices = np.argsort(semantic_scores)[::-1][:search_k]
        
        # Build results with hybrid scoring
        results = []
        for idx in top_indices:
            row = self.metadata_df.iloc[idx].copy()
            semantic_score = float(semantic_scores[idx])
            
            # Compute keyword score
            kw_score = 0.0
            if positive_keywords or negative_keywords:
                kw_score = self._keyword_score(
                    row, 
                    positive_keywords or [], 
                    negative_keywords if not strict_negative_filter else None
                )
            
            # Check strict negative filter
            if strict_negative_filter and negative_keywords:
                cell_name = row['cell_type_name'].lower()
                definition = str(row.get('onto_definition', '')).lower()
                text = f"{cell_name} {definition}"
                
                has_negative = any(nk.lower() in text for nk in negative_keywords)
                if has_negative:
                    continue  # Skip this result entirely
            
            # Hybrid score
            hybrid_score = (1 - keyword_weight) * semantic_score + keyword_weight * (kw_score + 1) / 2
            
            row['semantic_score'] = semantic_score
            row['keyword_score'] = kw_score
            row['hybrid_score'] = hybrid_score
            row['similarity'] = hybrid_score  # For compatibility
            
            results.append(row)
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("No results found. Try relaxing filters.")
            return results_df
        
        # Add ontology depth
        if 'ontology_depth' not in results_df.columns:
            results_df['ontology_depth'] = results_df['ontology_id'].apply(self._get_ontology_depth)
        
        # Apply filters
        if exclude_broad_terms:
            results_df = results_df[results_df['ontology_depth'] > 0]
        
        if min_specificity > 0:
            results_df = results_df[results_df['ontology_depth'] >= min_specificity]
        
        if organ_filter:
            results_df = results_df[results_df['organ'] == organ_filter]
        
        if min_cells:
            results_df = results_df[results_df['n_cells'] >= min_cells]
        
        # Sort by hybrid score
        results_df = results_df.sort_values('hybrid_score', ascending=False).head(k)
        
        return results_df
    
    def search_gaba_neurons(self, 
                           organ: str = None, 
                           k: int = 10,
                           min_cells: int = 100,
                           include_interneurons: bool = True) -> pd.DataFrame:
        """
        Convenience method for finding GABAergic neurons
        Pre-configured with appropriate keywords
        """
        positive = ["GABA", "GABAergic", "inhibitory"]
        if include_interneurons:
            positive.append("interneuron")
        
        negative = ["glutamatergic", "glutamate", "excitatory"]
        
        query = "GABAergic inhibitory neuron GABA neurotransmitter"
        
        results = self.search_hybrid(
            query_text=query,
            positive_keywords=positive,
            negative_keywords=negative,
            keyword_weight=0.4,  # Higher weight for keywords
            organ_filter=organ,
            min_cells=min_cells,
            k=k,
            strict_negative_filter=True
        )
        
        self._display_results(results, f"GABAergic Neurons" + (f" in {organ}" if organ else ""))
        return results
    
    def search_glutamatergic_neurons(self,
                                     organ: str = None,
                                     k: int = 10,
                                     min_cells: int = 100) -> pd.DataFrame:
        """
        Convenience method for finding glutamatergic neurons
        """
        positive = ["glutamatergic", "glutamate", "excitatory"]
        negative = ["GABA", "GABAergic", "inhibitory"]
        
        query = "glutamatergic excitatory neuron glutamate neurotransmitter"
        
        results = self.search_hybrid(
            query_text=query,
            positive_keywords=positive,
            negative_keywords=negative,
            keyword_weight=0.4,
            organ_filter=organ,
            min_cells=min_cells,
            k=k,
            strict_negative_filter=True
        )
        
        self._display_results(results, f"Glutamatergic Neurons" + (f" in {organ}" if organ else ""))
        return results
    
    def search_with_exclusions(self,
                               query: str,
                               must_contain: List[str] = None,
                               must_not_contain: List[str] = None,
                               organ: str = None,
                               k: int = 10,
                               min_cells: int = 100) -> pd.DataFrame:
        """
        Search with explicit inclusion/exclusion criteria
        
        Parameters:
        -----------
        query : str
            Semantic query
        must_contain : List[str]
            Cell type name must contain at least one of these
        must_not_contain : List[str]
            Cell type name must NOT contain any of these
        
        Example:
        --------
        retriever.search_with_exclusions(
            query="inhibitory neurons",
            must_contain=["GABA"],
            must_not_contain=["glutamatergic", "excitatory"],
            organ="brain"
        )
        """
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Build results with filtering
        results = []
        for idx in np.argsort(similarities)[::-1]:
            row = self.metadata_df.iloc[idx].copy()
            cell_name_lower = row['cell_type_name'].lower()
            
            # Check must_contain
            if must_contain:
                if not any(mc.lower() in cell_name_lower for mc in must_contain):
                    continue
            
            # Check must_not_contain
            if must_not_contain:
                if any(mnc.lower() in cell_name_lower for mnc in must_not_contain):
                    continue
            
            # Apply other filters
            if organ and row['organ'] != organ:
                continue
            
            if min_cells and row['n_cells'] < min_cells:
                continue
            
            row['similarity'] = float(similarities[idx])
            results.append(row)
            
            if len(results) >= k:
                break
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            results_df['ontology_depth'] = results_df['ontology_id'].apply(self._get_ontology_depth)
        
        self._display_results(results_df, f"Search: {query[:50]}...")
        return results_df
    
    def _display_results(self, results_df: pd.DataFrame, title: str):
        """Display search results in a nice format"""
        print(f"\n{'='*100}")
        print(f"{title}")
        print(f"{'='*100}\n")
        
        if len(results_df) == 0:
            print("No results found.")
            return
        
        # Determine which score columns to show
        score_cols = []
        if 'hybrid_score' in results_df.columns:
            score_cols = ['semantic_score', 'keyword_score', 'hybrid_score']
        else:
            score_cols = ['similarity']
        
        print(f"{'Rank':<5} {'Cell Type':<45} {'Score':<10} {'N Cells':<10} {'Organ':<10}")
        print("-"*100)
        
        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            score = row.get('hybrid_score', row.get('similarity', 0))
            print(f"{rank:<5} {row['cell_type_name']:<45} {score:<10.3f} "
                  f"{row['n_cells']:<10,} {row['organ']:<10}")
            
            # Show detailed scores for hybrid search
            if 'hybrid_score' in results_df.columns:
                print(f"      └─ semantic: {row['semantic_score']:.3f}, "
                      f"keyword: {row['keyword_score']:.3f}")
        
        print()
    
    # ==================== ORIGINAL METHODS (kept for compatibility) ====================
    
    def search_semantic(self, 
                       query_text: str,
                       k: int = 10,
                       organ_filter: str = None,
                       min_cells: int = None,
                       exclude_broad_terms: bool = True,
                       min_specificity: int = 0) -> pd.DataFrame:
        """
        Pure semantic search - original method kept for compatibility
        
        NOTE: For better results with specific cell types, use search_hybrid() instead
        """
        # Embed query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search with embeddings
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        search_k = min(k * 10, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:search_k]
        
        # Build results
        results = []
        for idx in top_indices:
            row = self.metadata_df.iloc[idx].copy()
            row['similarity'] = float(similarities[idx])
            results.append(row)
        
        results_df = pd.DataFrame(results)
        
        # Add ontology depth
        if 'ontology_depth' not in results_df.columns:
            results_df['ontology_depth'] = results_df['ontology_id'].apply(self._get_ontology_depth)
        
        # Filters
        if exclude_broad_terms:
            results_df = results_df[results_df['ontology_depth'] > 0]
        
        if min_specificity > 0:
            results_df = results_df[results_df['ontology_depth'] >= min_specificity]
        
        if organ_filter:
            results_df = results_df[results_df['organ'] == organ_filter]
        
        if min_cells:
            results_df = results_df[results_df['n_cells'] >= min_cells]
        
        results_df = results_df.sort_values('similarity', ascending=False).head(k)
        
        return results_df
    
    def _get_ontology_depth(self, ontology_id: str) -> int:
        """Estimate cell type specificity by counting parent terms"""
        if ontology_id == 'unknown' or pd.isna(ontology_id):
            return 0
        
        matching_rows = self.metadata_df[self.metadata_df['ontology_id'] == ontology_id]
        
        if len(matching_rows) == 0:
            return 0
        
        row = matching_rows.iloc[0]
        
        if 'onto_parents' not in row or pd.isna(row['onto_parents']):
            return 0
        
        try:
            parents_json = row['onto_parents']
            if isinstance(parents_json, str):
                parents = json.loads(parents_json)
            else:
                parents = parents_json
            return len(parents) if isinstance(parents, list) else 0
        except (json.JSONDecodeError, TypeError):
            return 0
    
    def browse_available_celltypes(self,
                                   organ: str,
                                   min_cells: int = 100,
                                   min_specificity: int = 0,
                                   keyword: str = None,
                                   show_depth_distribution: bool = True):
        """Browse available cell types in the repository"""
        df = self.metadata_df[self.metadata_df['organ'] == organ].copy()
        df = df[df['n_cells'] >= min_cells]
        
        print(f"\n{'='*100}")
        print(f"Available cell types in {organ.upper()}")
        if keyword:
            print(f"  Keyword filter: '{keyword}'")
        print(f"  Before filters: {len(df)} cell types")
        print(f"{'='*100}")
        
        df['ontology_depth'] = df['ontology_id'].apply(self._get_ontology_depth)
        
        if show_depth_distribution:
            print("\nOntology depth distribution:")
            depth_counts = df['ontology_depth'].value_counts().sort_index()
            for depth, count in depth_counts.items():
                print(f"  Depth {depth}: {count} cell types")
        
        if min_specificity > 0:
            df = df[df['ontology_depth'] >= min_specificity]
            print(f"\nAfter specificity filter (depth >= {min_specificity}): {len(df)} cell types")
        
        if keyword:
            df = df[df['cell_type_name'].str.contains(keyword, case=False, na=False)]
            print(f"After keyword filter: {len(df)} cell types")
        
        df = df.sort_values('n_cells', ascending=False)
        
        print(f"\n{'Cell Type':<55} {'Dataset':<15} {'N Cells':<10} {'Depth':<7}")
        print("="*100)
        
        for _, row in df.head(20).iterrows():
            print(f"{row['cell_type_name']:<55} {row['dataset_id']:<15} "
                  f"{row['n_cells']:<10,} {row['ontology_depth']:<7}")
        
        if len(df) > 20:
            print(f"\n... and {len(df) - 20} more")
        
        return df
    def select_background_for_query(self, 
                                     query: str,
                                     organ_filter: str = None,
                                     min_cells: int = 100,
                                     max_results_per_type: int = 5) -> dict:
        """
        Automatically select background cell types based on ontology
        
        Parameters:
        -----------
        query : str
            Cell type name or CL ID (e.g., "GABAergic neuron" or "CL:0000617")
        organ_filter : str, optional
            Restrict to specific organ (e.g., 'brain')
        min_cells : int
            Minimum cells per cell type entry
        max_results_per_type : int
            Max entries to return per cell type
        
        Returns:
        --------
        dict with:
            - query_info: dict (query name, CL ID, etc.)
            - background_cell_types: list of dict (CL ID, name, n_datasets, n_cells)
            - available_entries: pd.DataFrame (actual metadata entries)
            - summary: dict (totals)
        
        Example:
        --------
        >>> result = retriever.select_background_for_query("GABAergic neuron", organ_filter="brain")
        >>> print(f"Found {len(result['background_cell_types'])} cell types")
        >>> print(f"Total cells: {result['summary']['total_cells']:,}")
        """
        if not self.ontology:
            raise ValueError("Ontology not loaded. Initialize with cl_obo_path parameter.")
        
        print(f"\n{'='*100}")
        print(f"SELECTING BACKGROUND FOR: {query}")
        print(f"{'='*100}\n")
        
        # Step 1: Try ontology-based selection
        ontology_spec = self.ontology.get_background_spec(query)
        
        if ontology_spec['strategy'] == 'semantic_needed':
            # Ontology failed - use semantic search
            print(f"⚠ {ontology_spec['reasoning']}")
            print(f"  → Using semantic search to find closest match...\n")
            
            semantic_results = self.search_semantic(
                query_text=query,
                k=5,
                organ_filter=organ_filter,
                min_cells=min_cells,
                exclude_broad_terms=True
            )
            
            if len(semantic_results) == 0:
                raise ValueError(f"No matches found for query '{query}'")
            
            # Use top match
            best_match = semantic_results.iloc[0]
            matched_cl_id = best_match['ontology_id']
            
            print(f"✓ Best match: {best_match['cell_type_name']} ({matched_cl_id})")
            print(f"  Similarity: {best_match['similarity']:.3f}\n")
            
            # Now get siblings for the matched term
            ontology_spec = self.ontology.get_background_spec(matched_cl_id)
        
        # Step 2: Get siblings from ontology
        print(f"📊 Ontology Analysis:")
        print(f"  Query: {ontology_spec['query_name']} ({ontology_spec['query_cl_id']})")
        print(f"  Parent: {ontology_spec.get('biological_parent_name', 'unknown')}")
        print(f"  Siblings: {ontology_spec['n_siblings']}\n")
        
        # Step 3: Find which siblings are available in metadata
        background_cl_ids = ontology_spec['background_cl_ids']
        
        # Filter metadata for these CL IDs
        available_df = self.metadata_df[
            self.metadata_df['ontology_id'].isin(background_cl_ids)
        ].copy()
        
        # Apply filters
        if organ_filter:
            available_df = available_df[available_df['organ'] == organ_filter]
        
        if min_cells:
            available_df = available_df[available_df['n_cells'] >= min_cells]
        
        # Limit results per cell type
        if max_results_per_type:
            available_df = available_df.groupby('ontology_id').head(max_results_per_type)
        
        # Sort by cell count
        available_df = available_df.sort_values('n_cells', ascending=False)
        
        # Step 4: Summarize by cell type
        cell_type_summary = []
        for cl_id in background_cl_ids:
            entries = available_df[available_df['ontology_id'] == cl_id]
            if len(entries) > 0:
                cell_type_summary.append({
                    'ontology_id': cl_id,
                    'cell_type_name': entries.iloc[0]['cell_type_name'],
                    'n_datasets': len(entries['dataset_id'].unique()),
                    'n_entries': len(entries),
                    'total_cells': entries['n_cells'].sum()
                })
        
        print(f"🔍 Available in Metadata:")
        print(f"  Found {len(cell_type_summary)}/{len(background_cl_ids)} sibling types")
        print(f"  Total entries: {len(available_df)}")
        print(f"  Total cells: {available_df['n_cells'].sum():,}\n")
        
        # Display top cell types
        if cell_type_summary:
            print(f"Top Background Cell Types:")
            print(f"{'Cell Type':<50s} {'Entries':<10s} {'Cells':<15s}")
            print("-"*80)
            for ct in sorted(cell_type_summary, key=lambda x: x['total_cells'], reverse=True)[:10]:
                print(f"{ct['cell_type_name']:<50s} {ct['n_entries']:<10d} {ct['total_cells']:>12,}")
            if len(cell_type_summary) > 10:
                print(f"... and {len(cell_type_summary)-10} more")
        
        return {
            'query_info': {
                'original_query': query,
                'resolved_cl_id': ontology_spec['query_cl_id'],
                'resolved_name': ontology_spec['query_name'],
                'strategy': ontology_spec['strategy']
            },
            'background_cell_types': cell_type_summary,
            'available_entries': available_df,
            'summary': {
                'n_sibling_types_ontology': len(background_cl_ids),
                'n_sibling_types_available': len(cell_type_summary),
                'n_entries': len(available_df),
                'total_cells': int(available_df['n_cells'].sum()),
                'datasets': available_df['dataset_id'].unique().tolist()
            }
        }
    
    def get_dataset_ids_for_background(self, background_result: dict) -> list:
        """
        Extract dataset IDs from background selection result
        
        Parameters:
        -----------
        background_result : dict
            Output from select_background_for_query()
        
        Returns:
        --------
        list of str: Dataset IDs to download
        """
        df = background_result['available_entries']
        return df['dataset_id'].unique().tolist()


class OntologyBackgroundSelector:
    """
    Selects background cell types based on Cell Ontology hierarchy
    """
    def __init__(self, obo_path: str):
        """
        Parameters:
        -----------
        obo_path : str
            Path to cl.obo file
        """
        self.terms = {}
        self.functional_keywords = ['secretory', 'signaling', 'responsive', 'active', 'capable']
        self._parse_obo(obo_path)
    
    def _parse_obo(self, obo_path: str):
        """Parse OBO file (simple parser, no pronto dependency)"""
        import re
        
        current_term = None
        
        with open(obo_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('[Term]'):
                    if current_term and 'id' in current_term:
                        self.terms[current_term['id']] = current_term
                    current_term = {}
                    
                elif current_term is not None:
                    if line.startswith('id: '):
                        current_term['id'] = line[4:].strip()
                    elif line.startswith('name: '):
                        current_term['name'] = line[6:].strip()
                    elif line.startswith('def: '):
                        current_term['definition'] = line[5:].strip()
                    elif line.startswith('is_a: '):
                        parent_match = re.match(r'is_a: (CL:\d+)', line)
                        if parent_match:
                            if 'parents' not in current_term:
                                current_term['parents'] = []
                            current_term['parents'].append(parent_match.group(1))
            
            # Add last term
            if current_term and 'id' in current_term:
                self.terms[current_term['id']] = current_term
    
    def _is_functional_parent(self, parent_id: str) -> bool:
        """Check if parent is functional classification"""
        if parent_id not in self.terms:
            return False
        parent_name = self.terms[parent_id]['name'].lower()
        return any(kw in parent_name for kw in self.functional_keywords)
    
    def get_biological_parents(self, cl_id: str) -> list:
        """Get biological parents (exclude functional)"""
        if cl_id not in self.terms:
            return []
        
        term = self.terms[cl_id]
        if 'parents' not in term:
            return []
        
        bio_parents = [p for p in term['parents'] if not self._is_functional_parent(p)]
        
        # Fallback: if no biological parents, return all
        if not bio_parents:
            bio_parents = term['parents']
        
        return bio_parents
    
    def get_children(self, cl_id: str) -> list:
        """Get direct children of a term"""
        children = []
        for tid, term in self.terms.items():
            if 'parents' in term and cl_id in term['parents']:
                children.append(tid)
        return children
    
    def get_siblings(self, cl_id: str, include_self: bool = False) -> list:
        """
        Get all siblings through biological parents
        
        Returns:
        --------
        list of str: Sibling CL IDs
        """
        bio_parents = self.get_biological_parents(cl_id)
        
        all_siblings = set()
        for parent_id in bio_parents:
            children = self.get_children(parent_id)
            all_siblings.update(children)
        
        if not include_self:
            all_siblings.discard(cl_id)
        
        return list(all_siblings)
    
    def find_term_by_name(self, query: str) -> str:
        """
        Find CL ID by cell type name (case-insensitive)
        
        Returns:
        --------
        str: CL ID or None
        """
        query_lower = query.lower()
        
        # Exact match first
        for tid, term in self.terms.items():
            if term.get('name', '').lower() == query_lower:
                return tid
        
        # Partial match
        for tid, term in self.terms.items():
            if query_lower in term.get('name', '').lower():
                return tid
        
        return None
    
    def get_background_spec(self, query: str) -> dict:
        """
        Get background specification for a query
        
        Parameters:
        -----------
        query : str
            Cell type name or CL ID
        
        Returns:
        --------
        dict with keys:
            - query_cl_id: str
            - query_name: str
            - strategy: 'ontology' or 'semantic_needed'
            - background_cl_ids: list of str
            - biological_parent: str
            - reasoning: str
        """
        # Check if it's already a CL ID
        if query.startswith('CL:'):
            cl_id = query
            if cl_id not in self.terms:
                return {
                    'query_cl_id': query,
                    'query_name': None,
                    'strategy': 'semantic_needed',
                    'background_cl_ids': [],
                    'reasoning': f'CL ID {query} not found in ontology'
                }
            query_name = self.terms[cl_id].get('name', query)
        else:
            # Try to find by name
            cl_id = self.find_term_by_name(query)
            if not cl_id:
                return {
                    'query_cl_id': None,
                    'query_name': query,
                    'strategy': 'semantic_needed',
                    'background_cl_ids': [],
                    'reasoning': f'No exact match for "{query}" - semantic search needed'
                }
            query_name = query
        
        # Get siblings
        siblings = self.get_siblings(cl_id, include_self=False)
        bio_parents = self.get_biological_parents(cl_id)
        
        parent_name = self.terms[bio_parents[0]].get('name', 'unknown') if bio_parents else 'unknown'
        
        return {
            'query_cl_id': cl_id,
            'query_name': query_name,
            'strategy': 'ontology',
            'background_cl_ids': siblings,
            'biological_parent': bio_parents[0] if bio_parents else None,
            'biological_parent_name': parent_name,
            'n_siblings': len(siblings),
            'reasoning': f'Found {len(siblings)} siblings through parent "{parent_name}"'
        }