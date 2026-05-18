from src.models.model_factory import ModelFactory
import yaml

# Đọc cấu hình
with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Khởi tạo mô hình tự xây
model = ModelFactory.build_custom_cnn(
    img_size=config['data']['img_size'],
    num_classes=len(config['classes'])
)

# Hiển thị cấu trúc mô hình
model.summary()