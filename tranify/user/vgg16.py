import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from django.conf import settings
from tensorflow.keras.optimizers import Adam
from .graph import plot_training_results, plot_confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def train_vgg16(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (224, 224)
    
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Rotates images up to 30 degrees
    width_shift_range=0.2,  # Shifts image width up to 20%
    height_shift_range=0.2, # Shifts image height up to 20%
    shear_range=0.2,
    zoom_range=0.3,         # Increased zoom range
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], # Random brightness changes
    fill_mode='nearest',
    validation_split=0.2
)

    train_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    validation_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-4:]:  # Unfreeze last 4 layers
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
    optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

    history = model.fit(train_data, epochs=epochs, validation_data=validation_data)
  
    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_vgg16.h5')

    model.save(model_path)
    name_model='VGG16'
    plot_training_results(history,name_model,user_id,project_name)
    plot_confusion_matrix(model, validation_data,name_model,user_id,project_name)
    
    return model_path, history

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from django.conf import settings
from .graph import plot_training_results, plot_confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def train_cnn(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (64, 64)
    
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      # Rotates images up to 20 degrees
        width_shift_range=0.2,  # Shifts image width up to 20%
        height_shift_range=0.2, # Shifts image height up to 20%
        shear_range=0.2,
        zoom_range=0.2,         # Zoom range
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Load training and validation data
    train_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='sparse', subset='training')
    validation_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='sparse', subset='validation')

    # Build the CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # To prevent overfitting
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for classification

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(train_data, epochs=epochs, validation_data=validation_data)

    # Save the trained model
    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_cnn.h5')
    model.save(model_path)

    # Plot training results and confusion matrix
    name_model = 'CNN'
    plot_training_results(history, name_model, user_id, project_name)
    plot_confusion_matrix(model, validation_data, name_model, user_id, project_name)

    return model_path, history
