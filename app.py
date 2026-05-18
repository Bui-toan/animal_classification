import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

# 1. Cấu hình giao diện
st.set_page_config(page_title="Hệ thống Nhận diện Đa Vật Thể", layout="wide")
st.title("🐾 Hệ thống Khoanh vùng & Nhận diện Động vật AI")

if 'history' not in st.session_state:
    st.session_state.history = []

# 2. Tải các mô hình
@st.cache_resource
def load_models():
    detector = YOLO('yolov8n.pt') 
    classifier = tf.keras.models.load_model('models_saved/best_animal_model.keras')
    return detector, classifier

detector, classifier = load_models()
classes = ["cat", "chicken", "dog", "pig"]
ANIMAL_IDS = [14, 15, 16, 17, 18, 19, 20] # Mã quét các loài động vật của YOLO

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Tải ảnh có một hoặc nhiều con vật lên...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert('RGB')
        st.image(original_image, caption='Ảnh gốc đầu vào', use_container_width=True)
        
        if st.button("Phân tích & Khoanh vùng ngay!"):
            results = detector(original_image, verbose=False)[0]
            boxes = results.boxes
            
            draw_img = original_image.copy()
            draw = ImageDraw.Draw(draw_img)
            
            # Danh sách tạm để lưu cấu trúc từng con vật quét được
            raw_detected_objects = []
            count = 1
            
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in ANIMAL_IDS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Cắt vùng để mô hình phân loại
                    cropped_zone = original_image.crop((x1, y1, x2, y2))
                    img_resized = cropped_zone.resize((224, 224))
                    img_array = np.expand_dims(np.array(img_resized), axis=0)
                    img_array = preprocess_input(img_array)
                    
                    pred = classifier.predict(img_array, verbose=False)
                    pred_class = classes[np.argmax(pred)]
                    confidence = float(np.max(pred) * 100)
                    
                    # Khoanh vùng và đánh số lên ảnh
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    draw.text((x1 + 8, y1 + 8), f"#{count}", fill="yellow")
                    
                    # Lưu thông tin thô vào list
                    raw_detected_objects.append({
                        "id": count,
                        "class": pred_class,
                        "confidence": confidence
                    })
                    count += 1
            
            if len(raw_detected_objects) == 0:
                st.warning("Không tìm thấy con vật nào trong ảnh!")
            else:
                st.image(draw_img, caption='Ảnh đã được khoanh vùng và đánh số', use_container_width=True)
                
                # --- THUẬT TOÁN GỘP NHÓM THEO LOÀI VẬT ---
                grouped_objects = {}
                for obj in raw_detected_objects:
                    animal_type = obj["class"]
                    if animal_type not in grouped_objects:
                        grouped_objects[animal_type] = []
                    grouped_objects[animal_type].append(obj)
                
                detected_list = []
                summary_list = []
                
                # Duyệt qua từng loài đã gộp nhóm để định dạng hiển thị
                for animal_type, objs in grouped_objects.items():
                    # Gom các số thứ tự lại (Ví dụ: [2, 3] thành "2, 3")
                    ids_str = ", ".join([str(o["id"]) for o in objs])
                    # Tính độ tin cậy trung bình của nhóm này
                    avg_confidence = sum([o["confidence"] for o in objs]) / len(objs)
                    
                    detail_text = f"🎯 **Vùng {ids_str}**: {animal_type} (Độ tin cậy TB: {avg_confidence:.1f}%)"
                    detected_list.append(detail_text)
                    summary_list.append(f"Vùng {ids_str}: {animal_type}")
                
                # Hiển thị kết quả đã gộp ra giao diện dưới ảnh
                st.markdown("### 📊 Kết quả phân tích chi tiết (Đã gộp nhóm):")
                for detail in detected_list:
                    st.info(detail)
                
                # Lưu vào lịch sử
                st.session_state.history.append({
                    "image": draw_img,
                    "result": " | ".join(summary_list),
                    "details": detected_list
                })

with col2:
    st.subheader("📜 Lịch sử quét nâng cao")
    if st.button("🗑️ Xóa sạch lịch sử"):
        st.session_state.history = []
        st.rerun()

    for i, item in enumerate(reversed(st.session_state.history)):
        actual_index = len(st.session_state.history) - 1 - i
        with st.expander(f"Lần quét #{actual_index + 1}: {item['result']}"):
            st.image(item['image'], width=250)
            for detail in item['details']:
                st.write(detail)
            if st.button(f"🗑️ Xóa mục này", key=f"del_{actual_index}"):
                st.session_state.history.pop(actual_index)
                st.rerun()