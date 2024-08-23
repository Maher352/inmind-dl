# train_val_split.py

'''
This file is used to split the data into training and validation
It utilises the dataset to filter out the invalid npy files, and their corresponding image
And then it copies the split data into new folders to be used later

!!!This file is not fully functional, the saved images should be AUGMENTED first!!! 
'''

import albumentations as A
from dataset import BMWDataset
import random
from saving_data import save_data

# Define directories
images_dir = r'Project/data/images'
bboxes_dir = r'Project/data/bounding_boxes'
output_dir = r'Project/data/yolo_training'

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

# Shuffle indices and split manually
random.seed(0)
random.shuffle(dataset.valid_file_indices)

split_idx = int(0.95 * len(dataset.valid_file_indices))
train_indices = dataset.valid_file_indices[:split_idx]
val_indices = dataset.valid_file_indices[split_idx:]


# Save training and validation data
save_data(images_dir, bboxes_dir, output_dir, train_indices, 'train')
save_data(images_dir, bboxes_dir, output_dir, val_indices, 'validate')

print(f"Training and validation data saved successfully.")