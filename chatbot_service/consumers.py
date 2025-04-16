import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .chatbotFunctions import interactive_chatbot  # Import chatbot logic

class ChatbotConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.dialogue_state = {}  # Initialize dialogue state

    async def disconnect(self, close_code):
        pass  # Handle cleanup if needed

    async def receive(self, text_data):
        data = json.loads(text_data)
        user_input = data.get('message', '')

        # Call the chatbot logic
        response, self.dialogue_state = interactive_chatbot(user_input, self.dialogue_state)

        # Send response back to the client
        await self.send(json.dumps({'response': response}))
