from django.urls import path
from . import consumers


websocket_urlpatterns = [
    path("ws/chatbot/", consumers.ChatbotConsumerDLRAG.as_asgi()),
]