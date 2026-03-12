import logging
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JDBC_URL = f"jdbc:postgresql://{os.getenv('POSTGRES_HOST', 'postgres')}:"
f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'ecommerce')}"

JDBC_PROPS = {
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "driver": "org.postgresql.Driver",
}


def build_user_features(raw_events: SparkSession) -> SparkSession:
    return (
        raw_events.groupBy("visitor_id")
        .agg(
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias(
                "total_views"
            ),
            F.sum(F.when(F.col("event_type") == "addtocart", 1).otherwise(0)).alias(
                "total_addtocarts"
            ),
            F.sum(F.when(F.col("event_type") == "transaction", 1).otherwise(0)).alias(
                "total_transactions"
            ),
            F.countDistinct("item_id").alias("unique_items_interacted"),
        )
        .withColumn(
            "conversion_rate",
            F.when(
                F.col("total_views") > 0,
                F.col("total_transactions") / F.col("total_views"),
            ).otherwise(0),
        )
        .withColumn("updated_at", F.current_timestamp())
    )


def build_item_features(raw_events: SparkSession) -> SparkSession:
    return (
        raw_events.groupBy("item_id")
        .agg(
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias(
                "total_views"
            ),
            F.sum(F.when(F.col("event_type") == "addtocart", 1).otherwise(0)).alias(
                "total_addtocarts"
            ),
            F.sum(F.when(F.col("event_type") == "transaction", 1).otherwise(0)).alias(
                "total_transactions"
            ),
            F.countDistinct("visitor_id").alias("unique_visitors_interacted"),
        )
        .withColumn(
            "conversion_rate",
            F.when(
                F.col("total_views") > 0,
                F.col("total_transactions") / F.col("total_views"),
            ).otherwise(0),
        )
        .withColumn("updated_at", F.current_timestamp())
    )


def main() -> None:
    spark = (
        SparkSession.builder.appName("FeatureEngineering")
        .config("spark.jars", "/app/jars/postgresql-42.7.3.jar")
        .getOrCreate()
    )

    try:
        start_time = time.time()
        raw_events = spark.read.jdbc(JDBC_URL, "raw_events", properties=JDBC_PROPS)

        user_features = build_user_features(raw_events)
        item_features = build_item_features(raw_events)

        user_features.write.jdbc(
            JDBC_URL, "user_features", mode="overwrite", properties=JDBC_PROPS
        )
        item_features.write.jdbc(
            JDBC_URL, "item_features", mode="overwrite", properties=JDBC_PROPS
        )
        logger.info(
            f"Processed {user_features.count()} users,"
            f" {item_features.count()} items in {time.time() - start_time:.2f}s"
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
