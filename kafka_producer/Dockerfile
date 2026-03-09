FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY kafka_producer/ ./kafka_producer/

CMD ["python", "-m", "kafka_producer.producer"]
