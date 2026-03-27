import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, List

import uvicorn
from cache import get_cached, set_cached
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import csr_matrix

# Global variable to store model and mappings in memory
# to not load from disk each time

model_assets: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    """
    Handles startup and shutdown events,
    loads the trained model and ID mappings into memory on startup.
    """
    model_path = os.getenv("MODEL_PATH", "best_model.pkl")

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                model_assets["model"] = data["model"]
                model_assets["user_id_mapping"] = {
                    j: i for i, j in enumerate(data["user_id_mapping"])
                }
                model_assets["item_id_mapping"] = data["item_id_mapping"]

        except Exception as e:
            print(f"Failed to load model assets: {e}")

    else:
        print(f"Warning: Model file not found at {model_path}")

    yield
    model_assets.clear()


app = FastAPI(title="E-Commerce Recommendation API", lifespan=lifespan)


class RecommendRequest(BaseModel):
    visitor_id: int
    item_id: int
    n: int = 10


class ItemScore(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    visitor_id: int
    recommendations: List[ItemScore]


class SimilarityResponse(BaseModel):
    item_id: int
    similarities: List[ItemScore]


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": "model" in model_assets}


@app.get("/recommendations/{visitor_id}", response_model=RecommendResponse)
async def get_recommendations(visitor_id: int, n: int = 10) -> RecommendResponse:
    key = f"recommendations:{visitor_id}:{n}"
    cached_response = get_cached(key)
    if cached_response:
        return RecommendResponse.model_validate_json(cached_response)

    if "model" not in model_assets:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")

    user_to_id = model_assets["user_id_mapping"]
    user_id = user_to_id.get(visitor_id)

    if user_id is None:
        # Visitor was not seen during the training phase
        return RecommendResponse(visitor_id=visitor_id, recommendations=[])

    # Generate recomendations with the latent factors
    empty_user_items = csr_matrix((1, len(model_assets["item_id_mapping"])))
    ids, scores = model_assets["model"].recommend(
        user_id, user_items=empty_user_items, N=n
    )
    # user_items = None to not filter already viewed items

    # Map internal ALS indices back to original product item_ids
    recommendations = [
        ItemScore(item_id=int(model_assets["item_id_mapping"][i]), score=float(j))
        for i, j in zip(ids, scores)
    ]

    response = RecommendResponse(visitor_id=visitor_id, recommendations=recommendations)
    set_cached(key, response.model_dump_json())

    return response


@app.get("/item/{item_id}/similar", response_model=SimilarityResponse)
async def get_item_recommendations(item_id: int, n: int = 10) -> SimilarityResponse:
    key = f"similar:{item_id}:{n}"
    cached_response = get_cached(key)
    if cached_response:
        return SimilarityResponse.model_validate_json(cached_response)

    if "model" not in model_assets:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")

    item_to_id = {int(v): k for k, v in enumerate(model_assets["item_id_mapping"])}
    item_idx = item_to_id.get(item_id)

    if item_idx is None:
        raise HTTPException(status_code=404, detail="Item not found")
    # Generate recomendations with the latent factors
    ids, scores = model_assets["model"].similar_items(item_idx, N=n)

    # Map internal ALS indices back to original product item_ids
    similar = [
        ItemScore(item_id=int(model_assets["item_id_mapping"][i]), score=float(j))
        for i, j in zip(ids, scores)
    ]

    response = SimilarityResponse(item_id=item_id, similarities=similar)
    set_cached(key, response.model_dump_json())

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
