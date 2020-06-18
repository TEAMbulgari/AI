from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('cam/', views.cam, name = 'cam'),
    path('up/', views.up, name = 'testup'),
    # path('process/', views.process, name = 'process'),
]
