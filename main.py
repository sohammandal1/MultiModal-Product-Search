# main.py

import os
import shutil
import kagglehub
from scripts.embedding_generator import generate_embeddings
from scripts.index_builder import build_indices
from scripts.search_engine import SearchEngine
from scripts.config import DATA_DIR, IMAGE_DIR

def setup_project():
    """
    Runs the full data download, embedding, and indexing pipeline.
    """
    # 1. Download dataset
    print("--- Step 1: Downloading Dataset ---")
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        try:
            cached_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
            # The downloaded path is often a folder with the data inside it.
            # We copy the contents to our target DATA_DIR.
            shutil.copytree(cached_path, DATA_DIR, dirs_exist_ok=True)
            print(f"Dataset copied to: {DATA_DIR}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return
    else:
        print("Dataset directory already exists. Skipping download.")
        
    if not IMAGE_DIR.exists():
        print(f"Error: 'images' subdirectory not found in {DATA_DIR}")
        return

    # 2. Generate embeddings
    print("\n--- Step 2: Generating Embeddings ---")
    generate_embeddings()

    # 3. Build FAISS indices
    print("\n--- Step 3: Building FAISS Indices ---")
    build_indices()

    print("\n--- Project Setup Complete ---")


def run_demo(engine):
    """
    Demonstrates the capabilities of the search engine.
    """
    print("\n--- Running Demo Queries ---\n")

    # 1. Text Search
    print(">>> Text Search: 'red dress for evening'")
    hits = engine.search_text('red dress for evening', k=5)
    for h in hits:
        print(f"    - {h['productDisplayName']} (Score: {h['_score']:.4f})")
    
    # 2. Image Search (using a sample image from the dataset for demo)
    try:
        sample_image_path = engine.metadata.iloc[0]['image_path']
        print(f"\n>>> Image Search: Using sample image '{sample_image_path.name}'")
        with open(sample_image_path, 'rb') as f:
            image_bytes = f.read()
        hits = engine.search_image(image_bytes=image_bytes, k=5)
        print("    Original:", engine.metadata.iloc[0]['productDisplayName'])
        for h in hits:
            print(f"    - {h['productDisplayName']} (Score: {h['_score']:.4f})")
    except (FileNotFoundError, IndexError) as e:
        print(f"Could not perform image search demo: {e}")

    # 3. Recommendations
    print("\n>>> Recommendations for user 'user_10'")
    recs = engine.recommend_for_user('user_10', k=5)
    for r in recs:
        print(f"    - {r.get('productDisplayName', r.get('id'))}")

    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    # Run the setup pipeline first
    setup_project()
    
    # Initialize the search engine and run the demo
    try:
        search_engine = SearchEngine()
        run_demo(search_engine)
    except Exception as e:
        print(f"\nAn error occurred while initializing the engine or running the demo: {e}")
        print("Please ensure all setup steps completed successfully.")