# E-Commerce Recommendation System

Recommendation engine built on the Retail Rocket dataset. The pipeline ingests behavioral events through Kafka, computes features with PySpark, trains an ALS collaborative filtering model, and serves recommendations via a FastAPI endpoint.
Everything runs in Docker Compose - one command to start the full stack.
The producer simulates real-time by replaying historical events from the dataset into Kafka. The consumer reads them and batch-inserts into PostgreSQL. From there, PySpark computes aggregated features (views, add-to-carts, purchases per user/item), and the ML pipeline trains an ALS model on weighted implicit feedback (views *1, add-to-cart *2, purchase *5). The trained model is served through FastAPI with Redis caching.

# Stack

Python 3.11, Apache Kafka 3.7 (KRaft mode, no ZooKeeper), PostgreSQL 15, PySpark, implicit library (ALS), MLFlow for experiment tracking, FastAPI + Uvicorn, Redis for caching, Docker Compose, GitHub Actions for CI.

# Project structure

kafka_producer/        – reads Retail Rocket dataset, publishes events to Kafka
kafka_consumer/        – consumes from Kafka, batch inserts into PostgreSQL
spark_processor/       – PySpark batch job: raw_events, user_features + item_features
ml_pipeline/           – train/test split, ALS training, MLFlow experiment logging
api_service/           – FastAPI serving: /recommendations, /item/similar, /health
feature_store/         – SQL migrations
tests/                 – unit tests for Pydantic models and API schemas
Running it
You'll need Docker with Compose v2 and the Retail Rocket events.csv file.
bash
git clone https://github.com/Fuad-grb/ecommerce-recsys.git
cd ecommerce-recsys

cp .env.example .env
# edit .env - set your postgres credentials

mkdir -p data
# put events.csv in data/

make up                              # starts kafka, postgres, redis, mlflow
docker compose run kafka_producer    # ingests events into kafka
docker compose run kafka_consumer    # writes to postgresql
docker compose run spark_processor   # builds user/item features
docker compose run ml_pipeline       # trains model + logs to mlflow
docker compose up -d api_service     # starts the API on port 8000

### Then:

### bash

curl http://localhost:8000/health

curl "http://localhost:8000/recommendations/286616?n=5"

curl "http://localhost:8000/item/76196/similar?n=5"

MLFlow UI is at http://localhost:5001.

## API

GET /health — status check, shows if model is loaded.

GET /recommendations/{visitor_id}?n=10 — top-N recommendations for a user. Returns empty list if the user wasn't in training data.

GET /item/{item_id}/similar?n=10 — items similar to the given one based on learned item factors. Returns 404 if item is unknown.
Results are cached in Redis with a 1-hour TTL.

## Model

ALS (Alternating Least Squares) from the implicit library. Trained on a user-item interaction matrix with weighted implicit scores. Train/test split is by timestamp (80/20), not random — this matters because in production you always predict the future from the past.
Tried three hyperparameter configs. Best result: precision@10 ≈ 2.8% with 150 factors, regularization 0.1, 50 iterations. This is expected for sparse implicit feedback — most users only have a handful of events.
Experiments are tracked in MLFlow with parameters, metrics, and model artifacts.

Some design choices
KRaft over ZooKeeper - Kafka 3.7 supports KRaft natively. One less service to run and configure. ZooKeeper is being phased out anyway.

Batch Spark instead of streaming - recommendations don't need to update on every click. Batch feature engineering runs in ~3.5 seconds across the full dataset. Streaming would add a lot of complexity (checkpointing, state stores, watermarks) for no real benefit here.

ALS over neural models - with sparse implicit data and no content features, matrix factorization is hard to beat. Trains in seconds, serves in milliseconds, no GPU needed. A two-tower or sequence model would make sense with richer data or at larger scale.
