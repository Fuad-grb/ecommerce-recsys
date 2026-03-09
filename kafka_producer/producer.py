import csv
import logging
import os
import time
from datetime import datetime

from confluent_kafka import KafkaError, Message, Producer

from kafka_producer.models import UserEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


KAFKA_SERVER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "user_events")
PRODUCER_DELAY = float(
    os.getenv("PRODUCER_DELAY", "0.005")
)  # Delay in seconds between messages
LOG_EVERY_N = int(os.getenv("LOG_EVERY_N", "100"))


def delivery_report(err: KafkaError | None, msg: Message) -> None:
    if err is not None:
        logger.error(f"Message delivery failed: {err}")


def run_producer() -> None:
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "events.csv")
    )
    conf = {"bootstrap.servers": KAFKA_SERVER, "client.id": "python-producer-v1"}
    try:
        producer = Producer(conf)
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        return
    try:
        with open(dataset_path, "r") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                raw_tid = row.get("transactionid")
                try:
                    event = UserEvent(
                        timestamp=datetime.fromtimestamp(
                            int(row["timestamp"]) / 1000
                        ),  # Convert milliseconds to seconds
                        visitor_id=int(row["visitorid"]),
                        event_type=row["event"],
                        item_id=int(row["itemid"]),
                        transaction_id=int(raw_tid) if raw_tid else None,
                    )
                    producer.produce(
                        TOPIC_NAME, event.to_json(), callback=delivery_report
                    )
                    count += 1
                    if count % LOG_EVERY_N == 0:
                        logger.info(f"Produced {count} messages")
                    time.sleep(PRODUCER_DELAY)
                    producer.poll(0)  # Trigger delivery report callbacks
                except Exception as e:
                    logger.error(f"Error while producing messages: {e}")

            producer.flush()
    except Exception as e:
        logger.error(f"Failed to read dataset or produce messages: {e}")


if __name__ == "__main__":
    run_producer()
