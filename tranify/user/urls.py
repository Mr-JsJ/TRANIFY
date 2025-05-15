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
  path('model_details/<int:model_id>/',views.model_details, name='model_details'),
  # path('test-model/<str:model_name>/', views.test_model, name='test_model'),
   path('test-model/<str:model_name>/<int:project_id>/', views.test_model, name='test_model'),
  path("delete-project/<int:project_id>/",views.delete_project, name="delete_project"),
  path("about/",views.about, name="about"),
  path("contact/",views.contact_view, name="contact"),
  path('get_progress/<int:user_id>/<str:project_name>/',views.get_training_progress, name='get_progress'),
]
if settings.DEBUG:  # Serve media files in development
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)