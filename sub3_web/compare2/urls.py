from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload2, name='upload2'),
    path('cam/', views.cam2, name = 'cam2'),
    path('up/', views.up2, name = 'testup2'),
    # path('process/', views.process, name = 'process'),
]
