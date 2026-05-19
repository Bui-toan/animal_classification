import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(config):
    img_size = config['data']['img_size']
    batch_size = config['data']['batch_size']
    proc_dir = config['paths']['processed_dir']

  
    train_datagen = ImageDataGenerator(
        rescale=1./255,          
        rotation_range=20,        
        width_shift_range=0.2,    
        height_shift_range=0.2,   
        horizontal_flip=True,     
        zoom_range=0.2            
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(proc_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(proc_dir, 'val'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen