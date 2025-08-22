import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Additional imports for data sanitization
from pathlib import Path
import numpy as np
import pandas as pd

# Import the core logic from your structured scripts
from scripts.search_engine import SearchEngine
from scripts.config import DEFAULT_TOP_K, HYBRID_SEARCH_ALPHA

# --- API SETUP ---
app = FastAPI(
    title="Multi-Modal Fashion Search API",
    description="API for searching fashion products using text, images, or a hybrid approach, plus user recommendations.",
    version="1.0.0"
)

# --- LOAD THE ENGINE ---
try:
    print("Loading the search engine... This may take a moment.")
    search_engine = SearchEngine()
    print("Search engine loaded successfully!")
except Exception as e:
    print(f"FATAL: Could not initialize SearchEngine: {e}")
    search_engine = None

# --- Pydantic Models for Request Bodies ---
class TextSearchRequest(BaseModel):
    q: str = Field(..., description="The text query to search for.", example="blue summer dress")
    k: int = Field(DEFAULT_TOP_K, gt=0, le=50, description="Number of results to return.")

# --- HELPER FUNCTION FOR JSON SERIALIZATION ---
def sanitize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts non-serializable types in search results to JSON-friendly formats.
    Specifically handles pathlib.Path, numpy numbers, and pandas' NA.
    """
    sanitized_list = []
    for hit in hits:
        sanitized_hit = {}
        for key, value in hit.items():
            if isinstance(value, Path):
                sanitized_hit[key] = str(value)
            elif isinstance(value, (np.integer, np.int64)):
                sanitized_hit[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sanitized_hit[key] = float(value)
            elif pd.isna(value):
                sanitized_hit[key] = None
            else:
                sanitized_hit[key] = value
        sanitized_list.append(sanitized_hit)
    return sanitized_list

# --- API ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    if search_engine is None:
        raise RuntimeError("Search Engine could not be initialized. API cannot start.")

@app.get("/", tags=["Status"])
async def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Fashion Search API!"}


@app.post("/search/text", tags=["Search"])
async def api_search_text(payload: TextSearchRequest):
    """
    Performs a search based on a text query.
    """
    try:
        hits = search_engine.search_text(query=payload.q, k=payload.k)
        sanitized_results = sanitize_hits(hits)
        return JSONResponse(content={'results': sanitized_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", tags=["Search"])
async def api_search_image(
    k: int = Form(DEFAULT_TOP_K),
    file: UploadFile = File(...)
):
    """
    Performs a search based on an uploaded image.
    """
    try:
        image_bytes = await file.read()
        hits = search_engine.search_image(image_bytes=image_bytes, k=k)
        sanitized_results = sanitize_hits(hits)
        return JSONResponse(content={'results': sanitized_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid", tags=["Search"])
async def api_search_hybrid(
    q: Optional[str] = Form(None),
    k: int = Form(DEFAULT_TOP_K),
    alpha: float = Form(HYBRID_SEARCH_ALPHA),
    file: Optional[UploadFile] = File(None)
):
    """
    Performs a hybrid search using both an optional image and optional text.
    - **alpha**: Weight of the image embedding (0.0 to 1.0).
                 1.0 = image-only, 0.0 = text-only.
    """
    if file is None and q is None:
        raise HTTPException(status_code=400, detail="You must provide either an image, text, or both.")

    try:
        image_bytes = await file.read() if file else None
        hits = search_engine.search_hybrid(text=q, image_bytes=image_bytes, k=k, alpha=alpha)
        sanitized_results = sanitize_hits(hits)
        return JSONResponse(content={'results': sanitized_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{user_id}", tags=["Recommendation"])
async def api_recommend(user_id: str, k: int = DEFAULT_TOP_K):
    """
    Recommends items for a given user ID based on their interaction history.
    """
    try:
        recs = search_engine.recommend_for_user(user_id=user_id, k=k)
        sanitized_results = sanitize_hits(recs)
        return JSONResponse(content={'results': sanitized_results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- To run the API server ---
# Command: uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
