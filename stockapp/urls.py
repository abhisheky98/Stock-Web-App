"""stockapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings

from django.urls import path, include
from stocks import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index),
    path('trend-graph/', views.trend_graph, name='trend_graph'),
    path('multiple-trend-graph/', views.multiple_trend_graph, name='multiple_trend_graph'),
    path('predict-stock/',views.predict_stock,name='predict_stock'),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

