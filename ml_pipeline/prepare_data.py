import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812


def prepare_data() -> None:
    spark = (
        SparkSession.builder.appName("Data Preparation")
        .config("spark.jars", "/app/jars/postgresql-42.7.3.jar")
        .getOrCreate()
    )
    url = "jdbc:postgresql://postgres:5432/ecommerce"

    properties = {
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "driver": "org.postgresql.Driver",
    }

    df = spark.read.jdbc(url=url, table="raw_events", properties=properties)
    df.show(5)

    df = df.filter(F.col("visitor_id").isNotNull() & F.col("item_id").isNotNull())

    weighted_df = df.withColumn(
        "event_weight",
        F.when(F.col("event_type") == "view", 1)
        .when(F.col("event_type") == "addtocart", 2)
        .when(F.col("event_type") == "transaction", 5)
        .otherwise(0),
    )

    # for quantile calculation,
    # since Spark's approxQuantile does not work with timestamp type
    weighted_df_with_double = weighted_df.withColumn(
        "ts_double", F.col("timestamp").cast("double")
    )

    cutoff = weighted_df_with_double.approxQuantile("ts_double", [0.8], 0.01)[
        0
    ]  # using 80% of the data for training and 20% for testing

    train_events = weighted_df_with_double.filter(F.col("ts_double") <= cutoff)
    test_events = weighted_df_with_double.filter(F.col("ts_double") > cutoff)

    train_data = train_events.groupBy("visitor_id", "item_id").agg(
        F.sum("event_weight").alias("score")
    )
    test_data = test_events.groupBy("visitor_id", "item_id").agg(
        F.sum("event_weight").alias("score")
    )

    train_data.write.jdbc(
        url=url, table="train_data", mode="overwrite", properties=properties
    )
    test_data.write.jdbc(
        url=url, table="test_data", mode="overwrite", properties=properties
    )


if __name__ == "__main__":
    prepare_data()
