import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(config):
    img_size = config['data']['img_size']
    batch_size = config['data']['batch_size']
    proc_dir = config['paths']['processed_dir']

    # Thuật toán tăng cường ảnh cho tập Train
    train_datagen = ImageDataGenerator(
        rescale=1./255,           # Chuẩn hóa pixel về [0, 1]
        rotation_range=20,        # Xoay ngẫu nhiên 20 độ
        width_shift_range=0.2,    # Dịch ngang
        height_shift_range=0.2,   # Dịch dọc
        horizontal_flip=True,     # Lật ảnh ngang
        zoom_range=0.2            # Phóng to/thu nhỏ
    )

    # Tập Val và Test chỉ chuẩn hóa, KHÔNG tăng cường để giữ tính khách quan
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