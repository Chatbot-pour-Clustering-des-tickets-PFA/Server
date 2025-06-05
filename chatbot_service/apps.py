from django.apps import AppConfig
from .start_conumer import start_kafka_consumer
import threading
import sys


class ChatbotServiceConfig(AppConfig):
    name = 'chatbot_service'

    def ready(self):
        if 'runserver' in sys.argv:
            print("⚙️ Starting Kafka consumer thread...")  # Debug message
            threading.Thread(target=start_kafka_consumer, daemon=True).start()

            print("✅ Kafka consumer thread started.")
