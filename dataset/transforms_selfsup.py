from torchvision import transforms
from config import cfg

def get_selfsup_transform():
    """自监督增强：更加强烈，适用于 Triplet / Contrastive."""
    return transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.RandomResizedCrop(cfg.input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.7]*3, [0.2]*3)
    ])
