FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn pydantic implicit pandas

COPY api_service/ ./api_service/

CMD ["python", "-m", "api_service.main"]
