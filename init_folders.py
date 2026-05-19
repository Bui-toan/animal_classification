import os

folders = [
    "data/raw",              
    "data/processed/train",  
    "data/processed/val",    
    "src/data_loader",       
    "src/models",            
    "src/training",          
    "models_saved",          
    "app"                    
]

classes = ["dog", "cat", "chicken", "pig"]

for folder in folders:
    if "train" in folder or "val" in folder:
        for cls in classes:
            os.makedirs(os.path.join(folder, cls), exist_ok=True)
    else:
        os.makedirs(folder, exist_ok=True)

print("--- Đã tạo xong bộ khung dự án! ---")