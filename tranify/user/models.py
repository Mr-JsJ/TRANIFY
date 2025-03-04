from django.db import models
from django.contrib.auth.models import User

class TrainedModel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="trained_models")  # Link to the user
    project_name = models.CharField(max_length=255)
    num_classes = models.IntegerField()
    class_names = models.JSONField(default=dict)  # Store class names as a JSON object
    image_counts = models.JSONField(default=dict)  # Store image counts per class as JSON
    epochs = models.IntegerField()
    trained_models = models.JSONField(default=dict)  # List of trained models (e.g., ["VGG16", "ResNet50"])
    model_paths = models.JSONField(default=dict)  # Dictionary of model names and their respective paths
    recall = models.JSONField(default=dict)  # Store recall scores per model
    f1_score = models.JSONField(default=dict)  # Store F1 scores per model
    precision = models.JSONField(default=dict)  # Store precision scores per model
    accuracy = models.JSONField(default=dict)  # Store accuracy scores per model
    loss = models.JSONField(default=dict)  # Store loss values per model
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp for when the model was trained

    def __str__(self):
        return f"{self.project_name} - {self.user.username}"


