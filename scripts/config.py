# scripts/config.py

from pathlib import Path
import torch

# --- DIRECTORIES ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset"
IMAGE_DIR = DATA_DIR / "images"
METADATA_CSV = DATA_DIR / "styles.csv"
EMBEDDING_DIR = BASE_DIR / "embeddings"

# Ensure embedding directory exists
EMBEDDING_DIR.mkdir(exist_ok=True)

# --- EMBEDDING & INDEX FILES ---
TEXT_EMB_PATH = EMBEDDING_DIR / "text_embeddings.npy"
IMG_EMB_PATH = EMBEDDING_DIR / "img_embeddings.npy"
TEXT_INDEX_PATH = EMBEDDING_DIR / "faiss_text.index"
IMG_INDEX_PATH = EMBEDDING_DIR / "faiss_img.index"
INTERACTION_PATH = EMBEDDING_DIR / "interactions.parquet"


# --- MODEL CONFIGURATION ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAIN = "laion2b_e16"

# --- EMBEDDING GENERATION ---
BATCH_SIZE = 64

# --- SEARCH & RECOMMENDATION ---
DEFAULT_TOP_K = 10
HYBRID_SEARCH_ALPHA = 0.5 # Default weight for image vs. text in hybrid search