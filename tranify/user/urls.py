from django.urls import path
from . import views
from .views import verify_otp
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
  path('register',views.signup_user,name='signup'),
  path("verify-otp/", verify_otp, name="verify_otp"),
  path('login',views.login_user,name='login'),
  path('',views.home,name='home'),
  path('logout_user',views.logout_user,name='logout'),
  path('upload',views.upload,name='upload'), 
  path('your-projects/',views.user_projects, name='your_projects'),
  path('model_details/<int:model_id>/',views.model_details, name='model_details')
  # path('process_training/', views.process_training, name='process_training')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
