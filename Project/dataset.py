# dataset_final.py

'''
This file can be used as a custom dataset class for the provided dataset
It takes the images and bounding boxes and converts them to tensor format after transforming them
the function visualize2D is commented out, it can be used to visualize the images with their bboxes after augmentation
'''

import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import matplotlib.pyplot as plt
from visualize import visualize2D

class BMWDataset(Dataset):
    def __init__(self, image_dir, bbox_dir, transform=None):
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        self.bbox_lables_dict =  {
            '0': {'class': 'forklift'}, 
            '1': {'class': 'rack'}, 
            '2': {'class': 'crate'}, 
            '3': {'class': 'floor'}, 
            '4': {'class': 'railing'}, 
            '5': {'class': 'pallet'}, 
            '6': {'class': 'stillage'}, 
            '7': {'class': 'iwhub'}, 
            '8': {'class': 'dolly'}
            }

        # List of file indices based on the shared suffix
        self.file_indices = [
            filename.split('_')[-1].split('.')[0] 
            for filename in os.listdir(self.image_dir) if filename.endswith('.png')
        ]

    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        # Get the current file index (e.g., '0000')
        file_index = self.file_indices[idx]

        image_path = os.path.join(self.image_dir, f"rgb_{file_index}.png")
        bbox_path = os.path.join(self.bbox_dir, f"bounding_box_2d_tight_{file_index}.npy")

        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load bounding boxes
        bboxes = np.load(bbox_path, allow_pickle=True)
        bboxes = np.array([list(item) for item in bboxes], dtype=np.float32)

        # Create a boolean mask for rows where the first element is not 3 -> to remove the floor bounding box
        mask = bboxes[:, 0] != 3
        filtered_bboxes = bboxes[mask]

        # If any transformations are provided, apply them to the image and bounding boxes
        if self.transform:
            # Filter out invalid bounding boxes
            valid_bboxes = []
            for _, x_min, y_min, x_max, y_max, _ in filtered_bboxes:
                if x_max > x_min and y_max > y_min:
                    valid_bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Apply transformations if there are valid bounding boxes
            if valid_bboxes:
                transformed = self.transform(image=image_rgb, bboxes=valid_bboxes)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                ### visualize2D(transformed_image, transformed_bboxes)

                transformed_image_np = np.array(transformed_image)
                transformed_image_tens = torch.tensor(transformed_image_np, dtype=torch.float32).permute(2, 0, 1)  # from HWC to CHW format

                transformed_bboxes = np.array([list(item) for item in transformed_bboxes], dtype=np.float32)
                transformed_bboxes_tens = torch.tensor(transformed_bboxes[:, 0:5], dtype=torch.float32)

                return transformed_image_tens, transformed_bboxes_tens

        # Convert image to tensor
        image_np = np.array(image)
        image_tens = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)  # from HWC to CHW format

        # Convert bboxes to tensors
        bboxes_tens = torch.tensor(filtered_bboxes[:, 0:5], dtype=torch.float32)

        return image_tens, bboxes_tens


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


# Example usage:
# Define directories
images_dir = r'Project/data/images'
bboxes_dir = r'Project/data/bounding_boxes'

# Define the transformation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Resize(height=512, width=512, p=1.0),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

# Create dataset
dataset = BMWDataset(images_dir, bboxes_dir, transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(dataset.__getitem__(398))

print('success')