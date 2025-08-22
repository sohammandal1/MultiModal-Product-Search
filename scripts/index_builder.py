# scripts/index_builder.py

import numpy as np
import faiss

from scripts.config import (
    TEXT_EMB_PATH, IMG_EMB_PATH, TEXT_INDEX_PATH, IMG_INDEX_PATH
)

def build_and_save_index(embedding_path, index_path):
    """Loads embeddings and builds a FAISS index."""
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    print(f"Loading embeddings from {embedding_path}...")
    embeddings = np.load(embedding_path)
    
    dim = embeddings.shape[1]
    # Using IndexFlatIP for cosine similarity on L2-normalized vectors
    index = faiss.IndexFlatIP(dim)
    
    print(f"Building FAISS index for {len(embeddings)} vectors...")
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to {index_path}")

def build_indices():
    """Builds and saves FAISS indices for both text and image embeddings."""
    print("Starting FAISS index building process...")
    if TEXT_INDEX_PATH.exists() and IMG_INDEX_PATH.exists():
        print("FAISS indices already exist. Skipping.")
        return
        
    build_and_save_index(TEXT_EMB_PATH, TEXT_INDEX_PATH)
    build_and_save_index(IMG_EMB_PATH, IMG_INDEX_PATH)
    print("All FAISS indices have been built and saved.")

if __name__ == '__main__':
    build_indices()