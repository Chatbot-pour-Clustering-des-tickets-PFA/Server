import json
from channels.generic.websocket import AsyncWebsocketConsumer

from .completeChatbot import interactive_chatbot, register_background_task  # n'oublie pas d'importer le register

class ChatbotConsumerDLRAG(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.dialogue_state = {}
        self.group_added = False
        

        # Join group for this user

    async def disconnect(self, close_code):
        if hasattr(self, 'user_id') and self.group_added:
            await self.channel_layer.group_discard(f"user_{self.user_id}", self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        user_input = data.get("message", "")
        model_type = data.get("model_type", "dl").lower()
        if not hasattr(self, 'user_id') and 'user_id' in data:
            self.user_id = data.get("user_id")
            print("userId", self.user_id)
            await self.channel_layer.group_add(f"user_{self.user_id}", self.channel_name)
            self.group_added = True


        response, self.dialogue_state,intent = await interactive_chatbot(
            user_input,
            dialogue_state=self.dialogue_state,
            user_id=self.user_id,
            model_type=model_type
        )

        await self.send(text_data=json.dumps({
            "response": response,
            "model_type": model_type
        }))

        

        # ✅ Enregistre le traitement lourd seulement pour les intents "error_report" et "feedback"
        # (car interactive_chatbot est déjà prévu pour ça)
        if model_type in ["dl", "rag"]:
            register_background_task(user_input, self.dialogue_state, self.user_id, model_type,intent)

    async def send_background_result(self, event):
        """Called when background processing completes"""
        response = event['response']
        print("we are in the consumer baby", response)
        await self.send(text_data=json.dumps({
            "response": response
        }))
