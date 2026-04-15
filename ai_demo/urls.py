from django.urls import path
from . import views

app_name = 'ai_demo'

urlpatterns = [
    path('image/', views.image_classify, name='image_classify'),
    path('api/classify/', views.classify_api, name='classify_api'),
    path('sentiment/', views.sentiment_analysis, name='sentiment_analysis'),
    path('api/sentiment/', views.sentiment_api, name='sentiment_api'),
]