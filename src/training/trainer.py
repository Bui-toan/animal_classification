import os
import yaml
import tensorflow as tf
from src.models.model_factory import ModelFactory
from src.data_loader.augmentor import get_data_generators

def train():
    # 1. Đọc cấu hình (nhớ thêm encoding='utf-8')
    with open("config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. Lấy bộ nạp ảnh (Data Generators)
    train_gen, val_gen = get_data_generators(config)

    # 3. Khởi tạo mô hình
    # Bạn có thể chọn ModelFactory.build_custom_cnn hoặc build_transfer_learning
    model = ModelFactory.build_transfer_learning(
        img_size=config['data']['img_size'],
        num_classes=len(config['classes'])
    )

    # 4. Biên dịch mô hình (Compile)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['hyperparameters']['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Thiết lập các chốt chặn (Callbacks) - Điểm cộng chuyên nghiệp cho nhóm
    if not os.path.exists(config['paths']['model_save_dir']):
        os.makedirs(config['paths']['model_save_dir'])

    callbacks = [
        # Tự động lưu mô hình có kết quả tốt nhất
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['paths']['model_save_dir'], 'best_animal_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        # Dừng sớm nếu AI không tiến bộ (chống Overfitting)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 6. BẮT ĐẦU HUẤN LUYỆN
    print("\n>>> Hệ thống bắt đầu dạy AI học. Vui lòng đợi...")
    history = model.fit(
        train_gen,
        epochs=config['hyperparameters']['epochs'],
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    print("\n>>> Huấn luyện hoàn tất! Model tốt nhất đã được lưu trong folder models_saved.")
    # Lưu history vào file để vẽ biểu đồ sau này
    import pickle
    with open('models_saved/train_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    return history

if __name__ == "__main__":
    train()