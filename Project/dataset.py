# dataset.py

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class BMWDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        # Load JSON files
        for json_file in os.listdir(self.json_dir):
            with open(os.path.join(self.json_dir, json_file)) as f:
                annotations = json.load(f)
                for key, value in annotations.items():
                    # Adjust the filename to include the correct prefix and extension
                    filename = f"rgb_{key.zfill(4)}.png"  # Assuming key is a number
                    label = value['class']
                    self.data.append((filename, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} not found")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self._label_to_tensor(label)

        return image, label

    def _label_to_tensor(self, label):
        label_map = {
            "forklift": 0,
            "rack": 1,
            "crate": 2,
            "floor": 3,
            "railing": 4,
            "pallet": 5,
            "stillage": 6,
            "iwhub": 7,
            "dolly": 8,
        }
        return torch.tensor(label_map[label.lower()], dtype=torch.long)
