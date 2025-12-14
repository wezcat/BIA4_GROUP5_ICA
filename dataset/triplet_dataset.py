import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletHistDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = root_dir; self.split = split; self.transform = transform
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {"benign": 0, "malignant": 1}
        self.images = []; self.labels = []; self._load_data()
    def _load_data(self):
        base_path = os.path.join(self.root_dir, self.split)
        for cls_name in self.classes:
            cls_path = os.path.join(base_path, cls_name)
            if not os.path.exists(cls_path): continue
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]; label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        positive_indices = [i for i, l in enumerate(self.labels) if l == label and i != idx]
        negative_indices = [i for i, l in enumerate(self.labels) if l != label]
        positive_idx = random.choice(positive_indices) if positive_indices else idx
        negative_idx = random.choice(negative_indices)
        pos_img = Image.open(self.images[positive_idx]).convert("RGB")
        neg_img = Image.open(self.images[negative_idx]).convert("RGB")
        if self.transform:
            pos_img = self.transform(pos_img); neg_img = self.transform(neg_img)
        return img, pos_img, neg_img, label, img_path
