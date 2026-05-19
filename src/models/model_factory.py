import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import MobileNetV2

class ModelFactory:
    @staticmethod
    def build_custom_cnn(img_size, num_classes):
       
        model = models.Sequential([
            layers.Input(shape=(img_size, img_size, 3)),
            
           
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
          
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
           
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5), 
            layers.Dense(num_classes, activation='softmax')
        ])
        return model

    @staticmethod
    def build_transfer_learning(img_size, num_classes):
       
        base_model = MobileNetV2(input_shape=(img_size, img_size, 3),
                                 include_top=False,
                                 weights='imagenet')
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model    