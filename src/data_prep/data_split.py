import os
import shutil
import random
import yaml

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_and_split():
    config = load_config()
    raw_dir = config['paths']['raw_dir']
    processed_dir = config['paths']['processed_dir']
    classes = config['classes']
    
    # Xóa dữ liệu cũ trong processed để làm mới hoàn toàn
    if os.path.exists(processed_dir):
        print(f"--- Đang tiến hành phân tách ảnh từ {raw_dir} vào {processed_dir} ---")
        print(f"--- Đã dọn dẹp thư mục {processed_dir} ---")

    for cls in classes:
        src_path = os.path.join(raw_dir, cls)
        if not os.path.exists(src_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy folder {cls} trong raw/")
            continue

        # Lọc lấy các file ảnh hợp lệ
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        # Tính toán mốc chia
        total = len(images)
        train_idx = int(total * config['data']['train_split'])
        val_idx = int(total * (config['data']['train_split'] + config['data']['val_split']))

        splits = {
            'train': images[:train_idx],
            'val': images[train_idx:val_idx],
            'test': images[val_idx:]
        }

        for split_name, file_list in splits.items():
            dest_dir = os.path.join(processed_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for f in file_list:
                shutil.copy(os.path.join(src_path, f), os.path.join(dest_dir, f))
        
        print(f"✅ {cls.upper()}: Tổng {total} ảnh -> Đã chia xong.")

if __name__ == "__main__":
    clean_and_split()