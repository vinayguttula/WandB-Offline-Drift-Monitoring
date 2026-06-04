from django.urls import path
from .views import predict, batch_drift

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('drift/', batch_drift, name='batch_drift'),
]
