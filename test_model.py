from src.models.model_factory import ModelFactory
import yaml

with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

model = ModelFactory.build_custom_cnn(
    img_size=config['data']['img_size'],
    num_classes=len(config['classes'])
)

model.summary()