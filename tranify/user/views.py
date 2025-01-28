from django.shortcuts import render, redirect
from django.contrib.auth import login,logout,authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from .otp import generate_otp, send_otp 
import os
from django.contrib.auth.decorators import login_required
from django.conf import settings


def signup_user(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        password = request.POST.get("password")
        cpassword = request.POST.get("cpassword")

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email is already registered. Please log in.")
            return redirect("signup")

        # Password confirmation check
        if password != cpassword:
            messages.error(request, "Passwords do not match.")
            return redirect("signup")

        # Generate OTP
        otp = generate_otp()

        # Send OTP via email
        if send_otp(email, otp):
            request.session["otp"] = otp  # Store OTP in session
            request.session["signup_data"] = {
                "name": name,
                "email": email,
                "password": password,
            }
            return redirect("verify_otp")  # Redirect to OTP verification page
        else:
            messages.error(request, "Failed to send OTP. Try again.")
            return redirect("signup")

    return render(request, "signup.html")


def verify_otp(request):
   
    if request.method == "POST":
        user_otp = request.POST.get("otp")
        stored_otp = request.session.get("otp")
        signup_data = request.session.get("signup_data")

        if user_otp == stored_otp and signup_data:
            # Create user account
            user = User.objects.create_user(
                username=signup_data["email"],
                first_name=signup_data["name"],
                email=signup_data["email"],
                password=signup_data["password"],
            )
            user.save()

            user_id = user.id
            user_folder_path = os.path.join(settings.MEDIA_ROOT,f'{user_id}-USER')
            # Ensure the directory is created if it doesn't exist
            os.makedirs(user_folder_path, exist_ok=True)

            # Authenticate and log in user
            new_user = authenticate(username=signup_data["email"], password=signup_data["password"])
            if new_user:
                login(request, new_user)
                del request.session["otp"]  # Clear session data after successful login
                del request.session["signup_data"]
                return redirect("home")
            else:
                messages.error(request, "Login failed. Try again.")
                return redirect("login")
        else:
            messages.error(request, "Invalid OTP. Try again.")
            return redirect("verify_otp")

    return render(request, "verify_otp.html")


def login_user(request):
    if request.POST:
        email=request.POST.get('email')
        password=request.POST.get('password')
        user = authenticate(username=email,password=password)
        if user:
            login(request,user)
            request.session['user_id'] = user.id
            return redirect('home')
        else:
            messages.error(request,"Ivalid credentials")
            return redirect('login')
    return render(request,'login.html')



def logout_user(request):
    logout(request)
    return redirect('home')

def home(request):
    return render(request,'home.html')


@login_required(login_url=login)
def upload(request):
    if request.method == "POST":
        project_name = request.POST.get("project_name") 
        no_of_classes = request.POST.get("no_of_classes")
        dataset_file = request.FILES.get("dataset")  # Getting the uploaded file
        epochs = request.POST.get("epochs")
        selected_models = request.POST.getlist("models")  # Getting selected models as a list

        if not project_name or not no_of_classes or not dataset_file or not epochs:
            messages.error(request, "All fields are required!")
            return redirect("upload")

    
        user_id = request.session.get('user_id')  
        if not user_id:
            messages.error(request, "User authentication required!")
            return redirect("login")  

        project_folder_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name)
        os.makedirs(project_folder_path, exist_ok=True)  

    
        dataset_path = os.path.join(project_folder_path, dataset_file.name)
        with open(dataset_path, "wb") as destination:
            for chunk in dataset_file.chunks():
                destination.write(chunk)

      
        project_data = {
            "project_name": project_name,
            "no_of_classes": no_of_classes,
            "epochs": epochs,
            "selected_models": selected_models,
            "dataset_path": dataset_path
        }
        print(project_data)
        # Display a success message
        messages.success(request, "Project classification submitted successfully!")

        return redirect("upload")  # Redirect to the upload page or another page

    return render(request, "upload.html")
