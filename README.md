

**Trainify** is a web-based machine learning platform designed to make image classification simple and accessible. It allows users to upload custom datasets, automatically detects classes, trains a fine-tuned VGG16 model, and provides real-time training feedback. You can also enable GPU acceleration using TensorFlow with CUDA and cuDNN (via Conda).

---

## ✅ Features

- Upload and auto-organize image datasets
- Class detection from dataset folder names
- Real-time training updates (via Django Channels)
- Optional GPU acceleration using CUDA/cuDNN (with conda)

---

## 🧪 Setup Instructions (Recommended with Miniconda)

### 1️⃣ Install Miniconda

Download and install from:  
👉 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

---

### 2️⃣ Create and Activate Environment

```bash
conda create -n trainify_env python=3.10
conda activate trainify_env
 
### 3️⃣ Install Python Dependencies

pip install -r requirements.txt

###4️⃣ Set Up Email OTP Authentication
Create a Gmail account.

Enable 2-Step Verification and generate an App Password.

Add the credentials in OTP.py:

EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"




IF YOU HAVE RTX 3050 or above

### Enable GPU Acceleration (Conda + CUDA + cuDNN)
You can run TensorFlow with GPU support without downloading CUDA manually using Conda.

✅ Recommended Versions for TensorFlow 2.12
Component	Version
TensorFlow	2.12
cudatoolkit	11.8
cuDNN	8.6

###1️⃣ Install CUDA Toolkit & cuDNN

conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6

2️⃣ Set Environment Variables

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib TF_FORCE_GPU_ALLOW_GROWTH=true
conda deactivate
conda activate trainify_env

3️⃣ Install GPU-Compatible TensorFlow

pip install tensorflow==2.12

4️⃣ Verify GPU Access

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


###5️⃣Run the Django Server

python manage.py runserver