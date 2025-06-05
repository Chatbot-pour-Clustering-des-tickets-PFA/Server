from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_interaction, name='chatbot_interaction'),
]
