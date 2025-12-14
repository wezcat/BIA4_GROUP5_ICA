import os
import json
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BreastHistDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = root_dir; self.split = split; self.transform = transform
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {"benign": 0, "malignant": 1}
        self.images = []; self.labels = []; self._load_data()
        self.mean, self.std = self._compute_stats()

    def _load_data(self):
        base_path = os.path.join(self.root_dir, self.split)
        for cls_name in self.classes:
            cls_path = os.path.join(base_path, cls_name)
            if not os.path.exists(cls_path): continue
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def _compute_stats(self) -> Tuple[List[float], List[float]]:
        """
        Load cached dataset mean/std if available; otherwise compute and cache them.
        Cache file: root_dir/dataset_stats.json
        """
        stats_path = os.path.join(self.root_dir, "dataset_stats.json")

        # Try to load cached stats
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data["mean"], data["std"]
            except Exception as e:
                print(f"Failed to read cached stats: {e}. Recomputing...")

        # Compute stats
        print("Computing dataset statistics (first run may take a few minutes)...")
        from tqdm import tqdm  # local import to avoid hard dependency if not used

        # NOTE: we avoid relying on external cfg here
        # If you want a custom size, pass a transform from outside instead.
        temp_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        mean = torch.zeros(3)
        std = torch.zeros(3)
        num_images = 0

        train_path = os.path.join(self.root_dir, "train")
        for cls_name in self.classes:
            cls_path = os.path.join(train_path, cls_name)
            if not os.path.exists(cls_path):
                continue

            for img_name in tqdm(os.listdir(cls_path), desc=f"Processing {cls_name}"):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(cls_path, img_name)
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = temp_transform(img)

                    mean += img_tensor.mean(dim=[1, 2])
                    std += img_tensor.std(dim=[1, 2])
                    num_images += 1

        if num_images > 0:
            mean /= num_images
            std /= num_images

        mean_list = mean.tolist()
        std_list = std.tolist()

        # Save stats
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump({"mean": mean_list, "std": std_list}, f)
            print(f"Dataset statistics saved to {stats_path}")
        except Exception as e:
            print(f"Failed to save dataset statistics: {e}")

        return mean_list, std_list

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path