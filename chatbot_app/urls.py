# chatbot_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chatbot_api, name='chatbot_api'),
]
