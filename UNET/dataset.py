# dataset2.py

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BMWDataset2(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_label = {
            (0, 0, 0, 0): "BACKGROUND",
            (25, 255, 82, 255): "iwhub",
            (25, 82, 255, 255): "dolly",
            (54, 255, 25, 255): "stillage",
            (140, 25, 255, 255): "crate",
            (140, 255, 25, 255): "rack",
            (226, 255, 25, 255): "floor",
            (255, 25, 197, 255): "pallet",
            (255, 111, 25, 255): "railing",
            (255, 197, 25, 255): "forklift",
            (0, 0, 0, 255): "UNLABELLED"
        }
        self.label_map = {v: k for k, v in enumerate(self.mask_label.values())}
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("rgb_", "semantic_segmentation_"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGBA"))

        # Initialize the label map for the mask
        mask_label = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

        # Map colors to labels
        for color, label in self.mask_label.items():
            mask_color = np.all(mask[:, :, :4] == np.array(color), axis=-1)
            mask_label[mask_color] = self.label_map[label]

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask_label)
            image = augmentations["image"]
            mask_label = augmentations["mask"]

        return image, mask_label