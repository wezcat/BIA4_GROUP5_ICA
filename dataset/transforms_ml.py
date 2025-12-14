from torchvision import transforms
from config import cfg

def get_ml_transform(feature_extractor_type: str):
    if feature_extractor_type in ["voting"]:
        return transforms.Compose([
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    return transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_STATS["mean"], std=IMAGENET_STATS["std"])
    ])