
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from django.conf import settings
from tensorflow.keras.optimizers import Adam
from .graph import plot_training_results, plot_confusion_matrix, calculate_metrics
from tensorflow.keras.callbacks import LambdaCallback
from .models import TrainingProgress

# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def clear_training_progress(user_id, project_name):
    TrainingProgress.objects.filter(user_id=user_id, project_name=project_name).delete()

def update_progress(user_id, project_name, epoch, loss, accuracy, model_name):
    TrainingProgress.objects.create(
        user_id=user_id,
        project_name=project_name,
        epoch=epoch,
        loss=loss,
        accuracy=accuracy,
        model_name=model_name
    )

# VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import os

def train_vgg16(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (224, 224)

    # ✅ Rescaling only
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # ✅ Load VGG16 base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # ✅ Unfreeze only last 4 layers (instead of 8)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # ✅ Add simplified classification head with L2 regularization and fewer neurons
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # Reduced from 256 to 128 + L2
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ✅ Callback to update training progress
    def on_epoch_end(epoch, logs):
        update_progress(user_id, project_name, epoch + 1, logs['loss'], logs['accuracy'], "VGG16")

    # ✅ Learning rate scheduler
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[
            LambdaCallback(on_epoch_end=on_epoch_end),
            lr_schedule
        ]
    )

    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_vgg16.h5')
    model.save(model_path)

    # ✅ Evaluation and visualization
    plot_training_results(history, "VGG16", user_id, project_name)
    plot_confusion_matrix(model, val_data, "VGG16", user_id, project_name)
    metrics = calculate_metrics(model, val_data, "VGG16", user_id, project_name)
    clear_training_progress(user_id, project_name)

    return model_path, history, metrics


# ResNet50
def train_resnet50(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (224, 224)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    val_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def on_epoch_end(epoch, logs):
        update_progress(user_id, project_name, epoch + 1, logs['loss'], logs['accuracy'], "ResNet50")

    history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_resnet50.h5')
    model.save(model_path)
    plot_training_results(history, "ResNet50", user_id, project_name)
    plot_confusion_matrix(model, val_data, "ResNet50", user_id, project_name)
    metrics = calculate_metrics(model, val_data, "ResNet50", user_id, project_name)
    clear_training_progress(user_id, project_name)

    return model_path, history, metrics

# MobileNetV2
def train_mobilenetv2(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (224, 224)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    val_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def on_epoch_end(epoch, logs):
        update_progress(user_id, project_name, epoch + 1, logs['loss'], logs['accuracy'], "MobileNetV2")

    history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_mobilenetv2.h5')
    model.save(model_path)
    plot_training_results(history, "MobileNetV2", user_id, project_name)
    plot_confusion_matrix(model, val_data, "MobileNetV2", user_id, project_name)
    metrics = calculate_metrics(model, val_data, "MobileNetV2", user_id, project_name)
    clear_training_progress(user_id, project_name)

    return model_path, history, metrics

# AlexNet
def train_alexnet(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (227, 227)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    val_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

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

    def on_epoch_end(epoch, logs):
        update_progress(user_id, project_name, epoch + 1, logs['loss'], logs['accuracy'], "AlexNet")

    history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_alexnet.h5')
    model.save(model_path)
    plot_training_results(history, "AlexNet", user_id, project_name)
    plot_confusion_matrix(model, val_data, "AlexNet", user_id, project_name)
    metrics = calculate_metrics(model, val_data, "AlexNet", user_id, project_name)
    clear_training_progress(user_id, project_name)

    return model_path, history, metrics

# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import LambdaCallback, ReduceLROnPlateau
# import os
# from django.conf import settings

# def train_vgg16(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
#     image_size = (224, 224)

#     # ✅ Only rescaling (no data augmentation)
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         validation_split=0.2
#     )

#     train_data = datagen.flow_from_directory(
#         dataset_path,
#         target_size=image_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         subset='training'
#     )

#     val_data = datagen.flow_from_directory(
#         dataset_path,
#         target_size=image_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         subset='validation'
#     )

#     # Load VGG16 base model
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#     # ✅ Unfreeze last 8 layers
#     for layer in base_model.layers[:-8]:
#         layer.trainable = False
#     for layer in base_model.layers[-8:]:
#         layer.trainable = True

#     # Add custom classification head
#     x = Flatten()(base_model.output)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=base_model.input, outputs=predictions)

#     model.compile(
#         optimizer=Adam(learning_rate=0.0001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     # Callback to update training progress
#     def on_epoch_end(epoch, logs):
#         update_progress(user_id, project_name, epoch + 1, logs['loss'], logs['accuracy'], "VGG16")

#     # ✅ Learning rate scheduler
#     lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

#     history = model.fit(
#         train_data,
#         epochs=epochs,
#         validation_data=val_data,
#         callbacks=[
#             LambdaCallback(on_epoch_end=on_epoch_end),
#             lr_schedule
#         ]
#     )

#     model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_vgg16.h5')
#     model.save(model_path)

#     # Visualization and evaluation
#     plot_training_results(history, "VGG16", user_id, project_name)
#     plot_confusion_matrix(model, val_data, "VGG16", user_id, project_name)
#     metrics = calculate_metrics(model, val_data, "VGG16", user_id, project_name)
#     clear_training_progress(user_id, project_name)

#     return model_path, history, metrics

# def train_vgg16(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
#     image_size = (224, 224)
#     datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
#     train_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
#     val_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     for layer in base_model.layers[-4:]:
#         layer.trainable = True

#     x = Flatten()(base_model.output)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#     def on_epoch_end(epoch, logs):
#         update_progress(user_id, project_name, epoch + 1, logs['loss'], logs['accuracy'], "VGG16")

#     history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

#     model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_vgg16.h5')
#     model.save(model_path)
#     plot_training_results(history, "VGG16", user_id, project_name)
#     plot_confusion_matrix(model, val_data, "VGG16", user_id, project_name)
#     metrics = calculate_metrics(model, val_data, "VGG16", user_id, project_name)
#     clear_training_progress(user_id, project_name)

#     return model_path, history, metrics