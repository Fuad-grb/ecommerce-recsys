import json
import logging
import os
import signal

import psycopg2
from confluent_kafka import Consumer, KafkaError
from psycopg2.extras import execute_values

from kafka_consumer.models import UserEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_SERVER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "user_events")
GROUP_ID = "event_processor_v1"

DB_CONF = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "database": os.getenv("POSTGRES_DB", "ecommerce"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

running = True


def signal_handler(sig: int, frame: object) -> None:
    global running
    logger.info("Shutdown signal received. Exiting...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def save_batch(conn: psycopg2.extensions.connection, batch: list[UserEvent]) -> None:
    if not batch:
        return
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO raw_events (
                    timestamp, visitor_id, event_type, item_id, transaction_id
                    )
                VALUES %s
                """,
                [
                    (
                        e.timestamp,
                        e.visitor_id,
                        e.event_type,
                        e.item_id,
                        e.transaction_id,
                    )
                    for e in batch
                ],
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save batch: {e}")


def run_consumer() -> None:
    global running
    conf = {
        "bootstrap.servers": KAFKA_SERVER,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC_NAME])
    conn = psycopg2.connect(**DB_CONF)
    batch: list[UserEvent] = []
    count = 0
    try:
        while running:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Consumer error: {msg.error()}")
                continue
            try:
                event_data = json.loads(msg.value().decode("utf-8"))
                event = UserEvent(**event_data)
                batch.append(event)
                count += 1
                if len(batch) >= BATCH_SIZE:
                    save_batch(conn, batch)
                    logger.info(
                        f"Saved batch of {len(batch)} events. Total processed: {count}"
                    )
                    batch.clear()
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        if batch:
            save_batch(conn, batch)
            logger.info(
                f"Saved final batch of {len(batch)} events. Total processed: {count}"
            )
        consumer.close()
        conn.close()


if __name__ == "__main__":
    run_consumer()
