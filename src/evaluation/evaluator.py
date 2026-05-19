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
    
    # 🔥 ĐIỂM SỬA CHỐT HẠ: Tắt tính năng trộn ảnh ngẫu nhiên để tránh lệch pha dữ liệu
    val_gen.shuffle = False
    val_gen.index_array = None 
    val_gen.reset() # Đưa con trỏ đọc ảnh về vị trí đầu tiên
    
    # 2. Tải mô hình tốt nhất đã lưu
    model = tf.keras.models.load_model('models_saved/best_animal_model.keras')
    
    # 3. Dự đoán trên tập Validation
    print("Đang tiến hành đánh giá mô hình (Vui lòng đợi trong giây lát)...")
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    # 4. Tính toán và Vẽ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Vẽ heatmap bảng màu xanh dương chuẩn công nghiệp
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Ma trận nhầm lẫn - Confusion Matrix (Đã đồng bộ nhãn)')
    
    # Tự động lưu bức ảnh ma trận đẹp đẽ này vào thư mục để chèn vào báo cáo Word
    plt.savefig('models_saved/confusion_matrix_report.png', bbox_inches='tight')
    print("-> Đã lưu ảnh ma trận mới tại: models_saved/confusion_matrix_report.png")
    plt.show()

    # 5. In báo cáo toán học chi tiết (Precision, Recall, F1-score) ra Terminal
    print("\n" + "="*50)
    print("BÁO CÁO CHI TIẾT CÁC CHỈ SỐ (PRECISION, RECALL, F1-SCORE):")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("="*50)

if __name__ == "__main__":
    evaluate_model()