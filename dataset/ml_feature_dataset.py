import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from dataset.supervised_dataset import BreastHistDataset
from dataset.transforms_ml import get_ml_transform
from dataset.handcrafted import extract_handcrafted_features_vector

class MLFeatureDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", feature_extractor_type: str = "handcrafted"):
        self.feature_extractor_type = feature_extractor_type
        #self.transform = get_ml_transform(feature_extractor_type)
        #self.dataset = BreastHistDataset(root_dir, split, transform=self.transform)
        if self.feature_extractor_type == "handcrafted":
            self.transform = None
        else:
            self.transform = get_ml_transform(feature_extractor_type)
        self.dataset = BreastHistDataset(root_dir, split, transform=self.transform)

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        img, label, path = self.dataset[idx]
        if self.feature_extractor_type == "handcrafted":
            # BreastHistDataset 在 transform=None 时，img 是 PIL.Image
            if isinstance(img, Image.Image):
                img_np = np.array(img.convert("RGB"))
            else:
                # 如果 transform 不小心被设置了，也兜底转成 numpy
                img_np = img.permute(1, 2, 0).numpy() if torch.is_tensor(img) else np.array(img)
            feat = extract_handcrafted_features_vector(img_np)  # np.ndarray[D]
        else:
            # 原始行为：flatten 像素
            feat = img.numpy().reshape(-1).astype(np.float32)

        return feat, label, path