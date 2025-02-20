
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from django.conf import settings

# def plot_training_results(history, name_model, user_id, project_name):
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Model Accuracy')
#     plt.legend()
#     plt.savefig(os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{name_model}_accuracy_plot.png'))
#     plt.close()

#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Model Loss')
#     plt.legend()
#     plt.savefig(os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{name_model}_loss_plot.png'))
#     plt.close()

# def plot_confusion_matrix(model, validation_data, name_model, user_id, project_name):
#     y_pred = np.argmax(model.predict(validation_data), axis=1)
#     y_true = validation_data.classes
#     cm = confusion_matrix(y_true, y_pred)

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=validation_data.class_indices.keys(), 
#                 yticklabels=validation_data.class_indices.keys())
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.savefig(os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{name_model}_confusion_matrix.png'))
#     plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from django.conf import settings

def plot_training_results(history, name_model, user_id, project_name):
    # Define directory where images will be saved
    save_dir = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name)
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{name_model} Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{name_model}_accuracy_plot.png'))
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{name_model} Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{name_model}_loss_plot.png'))
    plt.close()

def plot_confusion_matrix(model, validation_data, name_model, user_id, project_name):
    save_dir = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name)
    os.makedirs(save_dir, exist_ok=True)

    y_pred = np.argmax(model.predict(validation_data), axis=1)
    y_true = validation_data.classes
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=validation_data.class_indices.keys(), 
                yticklabels=validation_data.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name_model} Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f'{name_model}_confusion_matrix.png'))
    plt.close()
