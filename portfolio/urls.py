from django.urls import path
from . import views

app_name = 'portfolio'

urlpatterns = [
    path('', views.index, name='index'),
    path('project/<int:pk>/', views.project_detail, name='project_detail'),
]