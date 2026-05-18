import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import yaml

# 1. Load cấu hình
with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 2. Load model tốt nhất
model = tf.keras.models.load_model('models_saved/best_animal_model.keras')
class_names = ["cat", "chicken", "dog", "pig"]

def predict_image(img_path):
    # Tiền xử lý ảnh
    img = image.load_img(img_path, target_size=(224, 224)) # Thử tăng lên (299, 299)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Chuẩn hóa giống lúc train

    # Dự đoán
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    print(f"\nKết quả dự đoán: {class_names[np.argmax(predictions)]}")
    print(f"Độ tin cậy: {100 * np.max(predictions):.2f}%")

if __name__ == "__main__":
    test_path = "C:\\Users\\bui toan\\Downloads\\animal_classification\\cho_1.jpg" # Thay bằng tên file ảnh bạn vừa tải về
    predict_image(test_path)