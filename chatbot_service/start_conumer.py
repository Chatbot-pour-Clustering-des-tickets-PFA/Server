from confluent_kafka import Consumer, KafkaException
import threading
from .chatbotFunctions import determine_priority_with_emotion

def start_kafka_consumer():
    # Configuration for the Kafka Consumer
    config = {
        'bootstrap.servers': 'localhost:9092',  # Kafka broker address
        'group.id': 'my-group',                # Consumer group ID
        'auto.offset.reset': 'earliest'        # Start reading from the beginning
    }
    
    # Initialize the Consumer
    consumer = Consumer(config)
    topic = 'my-topic'
    consumer.subscribe([topic])

    print(f"Kafka Consumer subscribed to topic: {topic}")

    def consume_messages():
        try:
            while True:
                msg = consumer.poll(1.0)  # Poll for messages with a timeout
                if msg is None:
                    continue  # No message, continue polling
                if msg.error():
                    if msg.error().code() == KafkaException._PARTITION_EOF:
                        print(f"End of partition reached {msg.topic()} {msg.partition()}")
                    else:
                        print(f"Error: {msg.error()}")
                    continue
                print(f"Received message: {msg.value().decode('utf-8')}")
        except KeyboardInterrupt:
            print("Stopping Kafka Consumer...")
        finally:
            consumer.close()
   