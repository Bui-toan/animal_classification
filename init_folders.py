import os

# Danh sách các thư mục cần thiết
folders = [
    "data/raw",              # Nơi chứa ảnh gốc bạn tải về
    "data/processed/train",  # Ảnh dùng để dạy AI
    "data/processed/val",    # Ảnh dùng để kiểm tra AI
    "src/data_loader",       # Code để đọc ảnh
    "src/models",            # Code định nghĩa bộ não AI
    "src/training",          # Code thực hiện việc học
    "models_saved",          # Nơi cất giữ bộ não sau khi học xong
    "app"                    # Code làm giao diện
]

classes = ["dog", "cat", "chicken", "pig"]

for folder in folders:
    # Nếu là thư mục dữ liệu thì tạo thêm 4 thư mục con cho 4 loài vật
    if "train" in folder or "val" in folder:
        for cls in classes:
            os.makedirs(os.path.join(folder, cls), exist_ok=True)
    else:
        os.makedirs(folder, exist_ok=True)

print("--- Đã tạo xong bộ khung dự án! ---")