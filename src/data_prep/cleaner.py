import os
import cv2
from PIL import Image

def clean_animal_dataset(data_dir):
  
    print(f" Bắt đầu quá trình kiểm tra và làm sạch dữ liệu tại: {data_dir}")
    
    deleted_count = 0
    checked_count = 0
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
            
        print(f" Đang quét thư mục nhãn: {class_name}")
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            checked_count += 1
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    print(f" Ảnh lỗi cấu trúc dữ liệu pixel (Xóa): {img_name}")
                    os.remove(img_path)
                    deleted_count += 1
                    continue
                
                h, w, _ = cv_img.shape
                if h < 10 or w < 10:
                    print(f" Ảnh quá bé, không đủ đặc trưng trích xuất {w}x{h} (Xóa): {img_name}")
                    os.remove(img_path)
                    deleted_count += 1
                    
            except (IOError, SyntaxError) as e:
            
                print(f" File không phải định dạng ảnh chuẩn hoặc bị lỗi file (Xóa): {img_name}")
                if os.path.exists(img_path):
                    os.remove(img_path)
                deleted_count += 1

    print("-" * 50)
    print(f" KẾT QUẢ LÀM SẠCH:")
    print(f"   - Tổng số file đã kiểm tra: {checked_count}")
    print(f"   - Số lượng file lỗi đã tự động xóa: {deleted_count}")
    print(f"   - Tập dữ liệu hiện tại đạt trạng thái SẠCH 100%.")

if __name__ == "__main__":
    DATA_RAW_DIR = "data/raw" 
    
    if os.path.exists(DATA_RAW_DIR):
        clean_animal_dataset(DATA_RAW_DIR)
    else:
        print(f" Không tìm thấy thư mục {DATA_RAW_DIR}. Vui lòng kiểm tra lại đường dẫn!")