from django.urls import path
from . import views
from .views import verify_otp
urlpatterns = [
  path('register',views.signup_user,name='signup'),
  path("verify-otp/", verify_otp, name="verify_otp"),
  path('login',views.login_user,name='login'),
  path('',views.home,name='home'),
  path('logout_user',views.logout_user,name='logout'),
  path('upload',views.upload,name='upload'), 
]
