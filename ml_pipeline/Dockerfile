FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/jars && \
    curl https://jdbc.postgresql.org/download/postgresql-42.7.3.jar -o /app/jars/postgresql-42.7.3.jar



RUN pip install --no-cache-dir pyspark psycopg2-binary mlflow implicit sqlalchemy pandas numpy "scipy<1.12"

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    curl \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ml_pipeline/ ./ml_pipeline/

CMD ["sh", "-c", "python -m ml_pipeline.prepare_data && python -m ml_pipeline.train"]
