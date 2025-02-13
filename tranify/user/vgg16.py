import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from django.conf import settings
from tensorflow.keras.optimizers import Adam
from .graph import plot_training_results, plot_confusion_matrix

def train_vgg16(dataset_path, num_classes, user_id, project_name, epochs=10, batch_size=16):
    image_size = (224, 224)
    
    train_datagen = ImageDataGenerator(
       rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2
    )

    train_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
    validation_data = train_datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, epochs=epochs, validation_data=validation_data)
  
    model_path = os.path.join(settings.MEDIA_ROOT, f'{user_id}-USER', project_name, f'{project_name}_vgg16.h5')

    model.save(model_path)
    name_model='vgg16'
    plot_training_results(history,name_model,user_id,project_name)
    plot_confusion_matrix(model, validation_data,name_model,user_id,project_name)
    
    return model_path, history

