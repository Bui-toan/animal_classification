import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import yaml
from src.data_loader.augmentor import get_data_generators

def evaluate_model():
    # 1. Tải cấu hình và dữ liệu
    with open("config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    _, val_gen = get_data_generators(config)
    
    # 2. Tải mô hình tốt nhất đã lưu
    model = tf.keras.models.load_model('models_saved/best_animal_model.keras')
    
    # 3. Dự đoán trên tập Validation
    print("Đang đánh giá mô hình...")
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    # 4. Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Ma trận nhầm lẫn - Confusion Matrix')
    plt.show()

    # 5. In báo cáo chi tiết (Precision, Recall, F1)
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    evaluate_model()