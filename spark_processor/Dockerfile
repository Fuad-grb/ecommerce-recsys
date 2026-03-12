FROM python:3.11-slim

RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app

ADD https://jdbc.postgresql.org/download/postgresql-42.7.3.jar /app/jars/postgresql-42.7.3.jar

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY spark_processor/ ./spark_processor/

CMD ["python", "-m", "spark_processor.feature_engineering"]
