# saving_data.py

'''
This file contains a single function that can be used to copy data from 1 folder into another one
It is used for splitting the data into trainning and validation
'''

import os
from shutil import copy2

# Function to save data to train and validate folders
def save_data(images_dir, bboxes_dir, output_dir, split_indices, split_name):
    split_image_dir = os.path.join(output_dir, split_name, 'images')
    split_bbox_dir = os.path.join(output_dir, split_name, 'labels')

    os.makedirs(split_image_dir, exist_ok=True)
    os.makedirs(split_bbox_dir, exist_ok=True)

    for file_index in split_indices:
        image_filename = f"rgb_{file_index}.png"
        bbox_filename = f"bounding_box_2d_tight_{file_index}.npy"

        copy2(os.path.join(images_dir, image_filename), os.path.join(split_image_dir, image_filename))
        copy2(os.path.join(bboxes_dir, bbox_filename), os.path.join(split_bbox_dir, bbox_filename))