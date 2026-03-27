import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    if "model" not in model_assets:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")

    user_to_id = model_assets["user_id_mapping"]
    user_id = user_to_id.get(visitor_id)

    if user_id is None:
        # Visitor was not seen during the training phase
        return RecommendResponse(visitor_id=visitor_id, recommendations=[])

    # Generate recomendations with the latent factors
    ids, scores = model_assets["model"].recommend(user_id, user_items=None, N=n)
    # user_items = None to not filter already viewed items

    # Map internal ALS indices back to original product item_ids
    recommendations = [
        ItemScore(item_id=int(model_assets["item_id_mapping"][i]), score=float(j))
        for i, j in zip(ids, scores)
    ]

    return RecommendResponse(visitor_id=visitor_id, recommendations=recommendations)


@app.get("/item/{item_id}/similar", response_model=SimilarityResponse)
async def get_item_recommendations(item_id: int, n: int = 10) -> SimilarityResponse:
    if "model" not in model_assets:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")

    # Generate recomendations with the latent factors
    ids, scores = model_assets["model"].similar_items(item_id, N=n)

    # Map internal ALS indices back to original product item_ids
    similar = [
        ItemScore(item_id=int(model_assets["item_id_mapping"][i]), score=float(j))
        for i, j in zip(ids, scores)
    ]

    return SimilarityResponse(item_id=item_id, similarities=similar)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
