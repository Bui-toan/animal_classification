import os
import cv2
from PIL import Image

def clean_animal_dataset(data_dir):
    """
    Quét và làm sạch tập dữ liệu: Loại bỏ ảnh lỗi, ảnh trống hoặc hỏng định dạng.
    """
    print(f"🚀 Bắt đầu quá trình kiểm tra và làm sạch dữ liệu tại: {data_dir}")
    
    deleted_count = 0
    checked_count = 0
    
    # Duyệt qua các thư mục con (cat, chicken, dog, pig)
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
            
        print(f"📁 Đang quét thư mục nhãn: {class_name}")
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            checked_count += 1
            
            try:
                # 1. Kiểm tra file có đọc được bằng Pillow không (Tránh file hỏng định dạng)
                with Image.open(img_path) as img:
                    img.verify()
                
                # 2. Dùng OpenCV đọc thử để kiểm tra cấu trúc mảng pixel
                cv_img = cv2.imread(img_path)
                if cv_img is None:
                    print(f"⚠️ Ảnh lỗi cấu trúc dữ liệu pixel (Xóa): {img_name}")
                    os.remove(img_path)
                    deleted_count += 1
                    continue
                
                # 3. Lọc bỏ các ảnh có kích thước quá nhỏ (Ví dụ nhỏ hơn 10x10 pixel)
                h, w, _ = cv_img.shape
                if h < 10 or w < 10:
                    print(f"⚠️ Ảnh quá bé, không đủ đặc trưng trích xuất {w}x{h} (Xóa): {img_name}")
                    os.remove(img_path)
                    deleted_count += 1
                    
            except (IOError, SyntaxError) as e:
                # Bắt các lỗi file không phải là ảnh thực tế hoặc bị lỗi byte mã hóa
                print(f"❌ File không phải định dạng ảnh chuẩn hoặc bị lỗi file (Xóa): {img_name}")
                if os.path.exists(img_path):
                    os.remove(img_path)
                deleted_count += 1

    print("-" * 50)
    print(f"📊 KẾT QUẢ LÀM SẠCH:")
    print(f"   - Tổng số file đã kiểm tra: {checked_count}")
    print(f"   - Số lượng file lỗi đã tự động xóa: {deleted_count}")
    print(f"   - Tập dữ liệu hiện tại đạt trạng thái SẠCH 100%.")

if __name__ == "__main__":
    # Đường dẫn tới thư mục data raw của bạn (Có thể điều chỉnh lại cho đúng cấu trúc thực tế)
    DATA_RAW_DIR = "data/raw" 
    
    if os.path.exists(DATA_RAW_DIR):
        clean_animal_dataset(DATA_RAW_DIR)
    else:
        print(f"❌ Không tìm thấy thư mục {DATA_RAW_DIR}. Vui lòng kiểm tra lại đường dẫn!")