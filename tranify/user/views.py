from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth import login,logout,authenticate
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.files.storage import default_storage
from django.contrib.auth.decorators import login_required
from django.conf import settings
import os
import zipfile
import shutil
from .otp import generate_otp, send_otp 
from .models import TrainedModel
from .train import train_vgg16,train_alexnet, train_resnet50, train_mobilenetv2




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

@login_required(login_url='login')
def upload(request):
    if request.method == "POST":
        project_name = request.POST.get("project_name")
        dataset_file = request.FILES.get("dataset")
        epochs = int(request.POST.get("epochs", 10))
        selected_models = request.POST.getlist("models")  # List of selected models

        if not project_name or not dataset_file or not selected_models:
            messages.error(request, "All fields are required, and at least one model must be selected!")
            return redirect("upload")

        user_id = request.session.get('user_id')  
        if not user_id:
            messages.error(request, "User authentication required!")
            return redirect("login")  

        project_folder_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name)
        os.makedirs(project_folder_path, exist_ok=True)

        num_classes, class_names, class_image_counts, dataset_root = handle_uploaded_zip(dataset_file, project_folder_path)

        if num_classes == 0:
            messages.error(request, "Invalid dataset. No valid classes found.")
            return redirect("upload")

        trained_models = []
        model_paths = {}
        metrics_dict = {}  # Dictionary to store metrics for each model

        # Train models based on selection
        if "vgg16" in selected_models:
            model_path, _, metrics = train_vgg16(dataset_root, num_classes, user_id, project_name, epochs)
            trained_models.append("VGG16")
            model_paths["VGG16"] = model_path
            metrics_dict["VGG16"] = metrics  # Store metrics

        if "alexnet" in selected_models:
            model_path, _, metrics = train_alexnet(dataset_root, num_classes, user_id, project_name, epochs)
            trained_models.append("AlexNet")
            model_paths["AlexNet"] = model_path
            metrics_dict["AlexNet"] = metrics

        if "resnet50" in selected_models:
            model_path, _, metrics = train_resnet50(dataset_root, num_classes, user_id, project_name, epochs)
            trained_models.append("ResNet50")
            model_paths["ResNet50"] = model_path
            metrics_dict["ResNet50"] = metrics

        if "mobilenetv2" in selected_models:
            model_path, _, metrics = train_mobilenetv2(dataset_root, num_classes, user_id, project_name, epochs)
            trained_models.append("MobileNetV2")
            model_paths["MobileNetV2"] = model_path
            metrics_dict["MobileNetV2"] = metrics

        # Extract metrics from dictionary
        recall_scores = {model: metrics["recall"] for model, metrics in metrics_dict.items()}
        f1_scores = {model: metrics["f1_score"] for model, metrics in metrics_dict.items()}
        precision_scores = {model: metrics["precision"] for model, metrics in metrics_dict.items()}
        accuracy_scores = {model: metrics["accuracy"] for model, metrics in metrics_dict.items()}
        loss_values = {model: metrics["loss"] for model, metrics in metrics_dict.items()}

        # Save project details to database
        trained_model = TrainedModel.objects.create(
            user=request.user,
            project_name=project_name,
            num_classes=num_classes,
            class_names=class_names,
            image_counts=class_image_counts,
            epochs=epochs,
            trained_models=trained_models,
            model_paths=model_paths,
            recall=recall_scores,
            f1_score=f1_scores,
            precision=precision_scores,
            accuracy=accuracy_scores,
            loss=loss_values
        )
        
        messages.success(request, f"Models trained successfully! Saved to project '{project_name}'.")
        return redirect("model_details", model_id=trained_model.id)

    return render(request, "upload.html")



from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

def augment_and_save_images(class_path, class_name, required_images):
    """Augments images for a given class if they are below the required number."""
    print(f"Augmenting class '{class_name}' to {required_images} images.")
    
    existing_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    num_existing = len(existing_images)

    if num_existing == 0:
        print(f"No images found for class '{class_name}'. Skipping augmentation.")
        return

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    images = []
    for img_name in existing_images:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))  # Resize to match model input
        images.append(img)

    images = np.array(images)
    augment_count = required_images - num_existing
    generated = 0

    for batch in datagen.flow(images, batch_size=1, save_to_dir=class_path, save_prefix="aug", save_format="jpg"):
        generated += 1
        if generated >= augment_count:
            break  # Stop when we have enough augmented images

def handle_uploaded_zip(file, extract_path):
    """Extracts and processes the uploaded dataset ZIP file."""
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path, exist_ok=True)

    temp_zip_path = os.path.join(extract_path, file.name)
    with default_storage.open(temp_zip_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    try:
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        os.remove(temp_zip_path)
        return 0, {}, {}, extract_path

    os.remove(temp_zip_path)

    extracted_dirs = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    dataset_root = os.path.join(extract_path, extracted_dirs[0]) if len(extracted_dirs) == 1 else extract_path

    class_names = [name for name in os.listdir(dataset_root)
                   if os.path.isdir(os.path.join(dataset_root, name))
                   and any(f.endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in os.listdir(os.path.join(dataset_root, name)))]

    num_classes = len(class_names)
    class_image_counts = {cls: count_images(os.path.join(dataset_root, cls)) for cls in class_names}

    # Perform augmentation if any class has <5000 images
    for class_name, count in class_image_counts.items():
        if count < 5000:
            class_path = os.path.join(dataset_root, class_name)
            augment_and_save_images(class_path, class_name, 5000)  # Augment up to 5000 images

    class_image_counts = {cls: count_images(os.path.join(dataset_root, cls)) for cls in class_names}  # Recalculate after augmentation

    return num_classes, class_names, class_image_counts, dataset_root



def count_images(directory):
    return len([file for file in os.listdir(directory) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])



def user_projects(request):
    projects = TrainedModel.objects.filter(user=request.user)
    return render(request, 'yourproject.html', {'projects': projects})


def model_details(request, model_id):
    trained_model = get_object_or_404(TrainedModel, id=model_id, user=request.user)
    print("trained_model",trained_model)
    return render(request, 'model_details.html', {
        'trained_model': trained_model,
        'MEDIA_URL': settings.MEDIA_URL  # Pass MEDIA_URL to the template
    })
    


