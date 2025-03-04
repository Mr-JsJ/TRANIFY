import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model,Sequential
from django.conf import settings
from tensorflow.keras.optimizers import Adam
from .graph import plot_training_results, plot_confusion_matrix, calculate_metrics


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
    metrics=calculate_metrics(model, validation_data, name_model, user_id, project_name)
    return model_path, history,metrics


def train_resnet50(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (224, 224)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

    train_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    validation_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-4:]:  
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, epochs=epochs, validation_data=validation_data)

    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_resnet50.h5')
    model.save(model_path)

    plot_training_results(history, "ResNet50", user_id, project_name)
    plot_confusion_matrix(model, validation_data, "ResNet50", user_id, project_name)
    metrics=calculate_metrics(model, validation_data, "ResNet50", user_id, project_name)
    return model_path, history,metrics


def train_mobilenetv2(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (224, 224)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

    train_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    validation_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-4:]:  
        layer.trainable = True


    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, epochs=epochs, validation_data=validation_data)

    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_mobilenetv2.h5')
    model.save(model_path)

    plot_training_results(history, "MobileNetV2", user_id, project_name)
    plot_confusion_matrix(model, validation_data, "MobileNetV2", user_id, project_name)
    metrics=calculate_metrics(model, validation_data, "MobileNetV2", user_id, project_name)
    return model_path, history,metrics


def train_alexnet(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (227, 227)  

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )

    train_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    validation_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

    model = Sequential([
        tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding="same"),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, epochs=epochs, validation_data=validation_data)

    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_alexnet.h5')
    model.save(model_path)

    plot_training_results(history, "AlexNet", user_id, project_name)
    plot_confusion_matrix(model, validation_data, "AlexNet", user_id, project_name)
    metrics=calculate_metrics(model, validation_data, "AlexNet", user_id, project_name)
    return model_path, history,metrics