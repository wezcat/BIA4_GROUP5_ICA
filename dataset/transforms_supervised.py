from torchvision import transforms
from config import cfg
from dataset.supervised_dataset import BreastHistDataset

def get_supervised_transforms(is_train: bool, root_dir: str = None):
    t = [transforms.Resize((cfg.input_size, cfg.input_size))]
    if is_train:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
              transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2)]
    t.append(transforms.ToTensor())
    if root_dir:
        try:
            ds = BreastHistDataset(root_dir, "train")
            t.append(transforms.Normalize(mean=ds.mean, std=ds.std))
        except:
            t.append(transforms.Normalize(mean=[0.5]*3, std=[0.5]*3))
    else:
        t.append(transforms.Normalize(mean=[0.5]*3, std=[0.5]*3))
    return transforms.Compose(t)

IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
