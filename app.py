import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import tensorflow as tf
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

st.set_page_config(page_title="Hệ thống Nhận diện Đa Vật Thể", layout="wide")
st.title("🐾 Hệ thống Khoanh vùng & Nhận diện Động vật AI")

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_models():
    detector = YOLO('yolov8n.pt') 
    classifier = tf.keras.models.load_model('models_saved/best_animal_model.keras')
    return detector, classifier

detector, classifier = load_models()
classes = ["cat", "chicken", "dog", "pig"]
ANIMAL_IDS = [14, 15, 16, 17, 18, 19, 20]

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Tải ảnh có một hoặc nhiều con vật lên...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert('RGB')
        st.image(original_image, caption='Ảnh gốc đầu vào', use_container_width=True)
        
        if st.button("Phân tích hình ảnh !"):
            results = detector(original_image, verbose=False)[0]
            boxes = results.boxes
            
            draw_img = original_image.copy()
            draw = ImageDraw.Draw(draw_img)
            
            raw_detected_objects = []
            count = 1
            
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in ANIMAL_IDS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cropped_zone = original_image.crop((x1, y1, x2, y2))
                    img_resized = cropped_zone.resize((224, 224))
                    img_array = np.expand_dims(np.array(img_resized), axis=0)
                    img_array = preprocess_input(img_array)
                    
                    pred = classifier.predict(img_array, verbose=False)
                    pred_class = classes[np.argmax(pred)]
                    confidence = float(np.max(pred) * 100)
                    
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    draw.text((x1 + 8, y1 + 8), f"#{count}", fill="yellow")
                    
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
                
                grouped_objects = {}
                for obj in raw_detected_objects:
                    animal_type = obj["class"]
                    if animal_type not in grouped_objects:
                        grouped_objects[animal_type] = []
                    grouped_objects[animal_type].append(obj)
                
                detected_list = []
                summary_list = []
                
                for animal_type, objs in grouped_objects.items():
                    ids_str = ", ".join([str(o["id"]) for o in objs])
                    avg_confidence = sum([o["confidence"] for o in objs]) / len(objs)
                    
                    detail_text = f"🎯 **Vùng {ids_str}**: {animal_type} (Độ tin cậy TB: {avg_confidence:.1f}%)"
                    detected_list.append(detail_text)
                    summary_list.append(f"Vùng {ids_str}: {animal_type}")
                
                st.markdown("### 📊 Kết quả phân tích chi tiết (Đã gộp nhóm):")
                for detail in detected_list:
                    st.info(detail)
                
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