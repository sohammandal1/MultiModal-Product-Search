# scripts/embedding_generator.py

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import open_clip

from scripts.data_loader import load_and_clean_metadata
from scripts.config import (
    DEVICE, TEXT_MODEL_NAME, CLIP_MODEL, CLIP_PRETRAIN, BATCH_SIZE,
    TEXT_EMB_PATH, IMG_EMB_PATH, EMBEDDING_DIR
)

def l2_normalize(x: np.ndarray):
    """Normalizes a numpy array of embeddings."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / norms

def generate_embeddings():
    """
    Generates and saves text and image embeddings for the dataset.
    """
    if TEXT_EMB_PATH.exists() and IMG_EMB_PATH.exists():
        print("Embeddings already exist. Skipping generation.")
        return

    # 1. Load data
    metadata = load_and_clean_metadata()
    if metadata.empty:
        print("Metadata is empty. Cannot generate embeddings.")
        return

    # 2. Load models
    print(f"Loading models on device: {DEVICE}")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN, device=DEVICE
    )
    clip_model.to(DEVICE)

    # 3. Initialize arrays
    n_items = len(metadata)
    text_dim = text_model.get_sentence_embedding_dimension()
    image_dim = clip_model.visual.output_dim

    text_embeddings = np.zeros((n_items, text_dim), dtype=np.float32)
    img_embeddings = np.zeros((n_items, image_dim), dtype=np.float32)

    # 4. Generate embeddings in batches
    print("Generating embeddings...")
    for i in tqdm(range(0, n_items, BATCH_SIZE)):
        batch_df = metadata.iloc[i:i + BATCH_SIZE]
        
        # Text embeddings
        texts = batch_df['productDisplayName'].astype(str).tolist()
        t_emb = text_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        text_embeddings[i:i + len(batch_df)] = t_emb

        # Image embeddings
        images = []
        for p in batch_df['image_path']:
            try:
                img = Image.open(p).convert('RGB')
                images.append(clip_preprocess(img))
            except Exception:
                # Append a blank image tensor on error
                blank_image = Image.new('RGB', (224, 224), (255, 255, 255))
                images.append(clip_preprocess(blank_image))
        
        img_tensor = torch.stack(images).to(DEVICE)
        with torch.no_grad():
            i_emb = clip_model.encode_image(img_tensor).cpu().numpy()
        img_embeddings[i:i + len(batch_df)] = i_emb

    # 5. Normalize and save
    text_embeddings = l2_normalize(text_embeddings)
    img_embeddings = l2_normalize(img_embeddings)
    
    np.save(TEXT_EMB_PATH, text_embeddings)
    np.save(IMG_EMB_PATH, img_embeddings)
    
    print(f"Embeddings saved to {EMBEDDING_DIR}")

if __name__ == '__main__':
    generate_embeddings()