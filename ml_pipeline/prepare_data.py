import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812

spark = SparkSession.builder.appName("Data Preparation").getOrCreate()

url = "jdbc:postgresql://postgres:5432/ecommerce"

properties = {
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
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

user_item_matrix = weighted_df.groupBy("visitor_id", "item_id").agg(
    F.sum("event_weight").alias("score")
)

user_item_matrix.show(10)

cutoff = weighted_df.approxQuantile("timestamp", [0.8], 0.01)[0]

train_events = weighted_df.filter(F.col("timestamp") <= cutoff)
test_events = weighted_df.filter(F.col("timestamp") > cutoff)

train_data = train_events.groupBy("visitor_id", "item_id").agg(
    F.sum("event_weight").alias("score")
)
test_data = test_events.groupBy("visitor_id", "item_id").agg(
    F.sum("event_weight").alias("score")
)
