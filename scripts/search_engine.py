# scripts/search_engine.py

import numpy as np
import pandas as pd
import torch
import faiss
import open_clip
import random
from PIL import Image
from sentence_transformers import SentenceTransformer
from io import BytesIO

from scripts.config import *
from scripts.data_loader import load_and_clean_metadata
from scripts.embedding_generator import l2_normalize

class SearchEngine:
    def __init__(self):
        print("Initializing Search Engine...")
        self.device = DEVICE
        self._load_models()
        self._load_data()
        self._load_indices()
        self._load_interactions()
        print("Search Engine ready.")

    def _load_models(self):
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME, device=self.device)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAIN, device=self.device
        )

    def _load_data(self):
        self.metadata = load_and_clean_metadata()
        self.img_embeddings = np.load(IMG_EMB_PATH)

    def _load_indices(self):
        self.faiss_text = faiss.read_index(str(TEXT_INDEX_PATH))
        self.faiss_img = faiss.read_index(str(IMG_INDEX_PATH))

    def _format_results(self, scores, indices):
        hits = []
        for score, idx in zip(scores, indices):
            row = self.metadata.iloc[int(idx)].to_dict()
            row['_score'] = float(score)
            hits.append(row)
        return hits

    def search_text(self, query: str, k: int = DEFAULT_TOP_K):
        q_emb = self.text_model.encode([query], convert_to_numpy=True)
        q_emb = l2_normalize(q_emb)
        D, I = self.faiss_text.search(q_emb.astype(np.float32), k)
        return self._format_results(D[0], I[0])

    def search_image(self, image_bytes: bytes, k: int = DEFAULT_TOP_K):
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor).cpu().numpy()
        emb = l2_normalize(emb)
        D, I = self.faiss_img.search(emb.astype(np.float32), k)
        return self._format_results(D[0], I[0])

    def search_hybrid(self, text: str = None, image_bytes: bytes = None, k: int = DEFAULT_TOP_K, alpha: float = HYBRID_SEARCH_ALPHA):
        if text is None and image_bytes is None:
            return []

        combined_scores = np.zeros(len(self.metadata), dtype=np.float32)

        if text:
            q_emb = self.text_model.encode([text], convert_to_numpy=True)
            q_emb = l2_normalize(q_emb)
            D_text, I_text = self.faiss_text.search(q_emb.astype(np.float32), len(self.metadata))
            scores_text = np.zeros(len(self.metadata), dtype=np.float32)
            scores_text[I_text[0]] = D_text[0]
            combined_scores += (1 - alpha) * scores_text

        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.clip_model.encode_image(img_tensor).cpu().numpy()
            emb = l2_normalize(emb)
            D_img, I_img = self.faiss_img.search(emb.astype(np.float32), len(self.metadata))
            scores_img = np.zeros(len(self.metadata), dtype=np.float32)
            scores_img[I_img[0]] = D_img[0]
            combined_scores += alpha * scores_img
            
        topk_idx = np.argsort(-combined_scores)[:k]
        return self._format_results(combined_scores[topk_idx], topk_idx)
    
    def _load_interactions(self):
        if INTERACTION_PATH.exists():
            self.interactions = pd.read_parquet(INTERACTION_PATH)
        else:
            print("Simulating user interactions...")
            users = [f"user_{i}" for i in range(1, 201)]
            items = self.metadata['id'].astype(str).tolist()[:2000]
            rows = [{'user': u, 'item': random.choice(items), 'rating': 1.0} for u in users for _ in range(random.randint(5, 30))]
            self.interactions = pd.DataFrame(rows)
            self.interactions.to_parquet(INTERACTION_PATH)
        
        self.pop_recs = self.interactions.groupby('item')['rating'].sum().sort_values(ascending=False).index.tolist()

    def recommend_for_user(self, user_id: str, k: int = DEFAULT_TOP_K):
        if user_id not in self.interactions['user'].unique(): # Cold start
            top_items = self.pop_recs[:k]
            return [self.metadata[self.metadata['id'].astype(str) == it].to_dict('records')[0] for it in top_items]

        user_items = self.interactions[self.interactions['user'] == user_id]['item'].tolist()
        last_item_id = user_items[-1]
        
        try:
            item_idx = self.metadata[self.metadata['id'].astype(str) == last_item_id].index[0]
        except IndexError:
            return [self.metadata.iloc[int(i)].to_dict() for i in range(k)] # Fallback
            
        q_emb = self.img_embeddings[item_idx:item_idx+1]
        D, I = self.faiss_img.search(q_emb.astype(np.float32), k + 1)
        
        recs = []
        for idx in I[0]:
            if int(idx) == item_idx: continue
            recs.append(self.metadata.iloc[int(idx)].to_dict())
            if len(recs) >= k: break
        return recs