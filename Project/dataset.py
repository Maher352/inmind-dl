# dataset_final.py

'''
This file can be used as a custom dataset class for the provided dataset
It takes the images and bounding boxes and converts them to tensor format after transforming them
the function visualize2D is commented out, it can be used to visualize the images with their bboxes after augmentation
'''

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class BMWDataset(Dataset):
    def __init__(self, image_dir, bbox_dir, transform=None):
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        self.bbox_labels_dict =  {
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

        # List of valid file indices based on the shared suffix
        self.valid_file_indices = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                file_index = filename.split('_')[-1].split('.')[0]
                bbox_path = os.path.join(self.bbox_dir, f"bounding_box_2d_tight_{file_index}.npy")
                bboxes = np.load(bbox_path, allow_pickle=True)
                bboxes = np.array([list(item) for item in bboxes], dtype=np.float32)

                if len(bboxes) == 0:
                    continue

                mask = bboxes[:, 0] != 3
                filtered_bboxes = bboxes[mask]

                if len(filtered_bboxes) == 0:
                    continue

                valid_bboxes = []
                for bbox_semantic_id, x_min, y_min, x_max, y_max, _ in filtered_bboxes:
                    if x_max > x_min and y_max > y_min:
                        valid_bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

                if valid_bboxes:
                    self.valid_file_indices.append(file_index)

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        file_index = self.valid_file_indices[idx]

        image_path = os.path.join(self.image_dir, f"rgb_{file_index}.png")
        bbox_path = os.path.join(self.bbox_dir, f"bounding_box_2d_tight_{file_index}.npy")

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = np.load(bbox_path, allow_pickle=True)
        bboxes = np.array([list(item) for item in bboxes], dtype=np.float32)

        mask = bboxes[:, 0] != 3
        filtered_bboxes = bboxes[mask]

        valid_bboxes = []
        for bbox_semantic_id, x_min, y_min, x_max, y_max, _ in filtered_bboxes:
            if x_max > x_min and y_max > y_min:
                valid_bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
                if (idx) == 10:
                    print(valid_bboxes)

        # THE BBOXES ARE NOT CLEAN, MANY BBOXES ARE TOO SMALL AND SOME BBOXES ARE REPEATED WITH SLIGHT SHIFTS 
        # TO PROPERLY LOAD THE BBOXES, THE REPEATED AND EXTREMLY SMALL BOXES MUST BE REMOVED
        # I'm running out of time but this is a very importent point that should've been done before training


        if self.transform:
            transformed = self.transform(image=image_rgb, bboxes=valid_bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            transformed_image_np = np.array(transformed_image)
            transformed_image_tens = torch.tensor(transformed_image_np, dtype=torch.float32).permute(2, 0, 1)

            transformed_bboxes_np = np.array([list(item) for item in transformed_bboxes], dtype=np.float32)
            transformed_bboxes_tens = torch.tensor(transformed_bboxes_np[:, 0:5], dtype=torch.float32)

            return transformed_image_tens, transformed_bboxes_tens

        image_np = np.array(image)
        image_tens = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)

        bboxes_tens = torch.tensor(valid_bboxes[:, 0:5], dtype=torch.float32)

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