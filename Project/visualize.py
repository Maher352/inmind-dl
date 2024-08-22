# visualize.py
'''
This file can be used to visualize all the images in the directory with their bounding boxes
It can also visualize them after transformation
Finally it has the visualize2D function which can be used in the dataloader
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A


def draw_with_bounding_boxes(image_dir, bbox_dir, limit=10):
    # Get list of image files and bbox files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    bbox_files = sorted([f for f in os.listdir(bbox_dir) if f.endswith('.npy')])
    
    # Check if number of images and bbox files match
    if len(image_files) != len(bbox_files):
        print("The number of images and bounding box files do not match.")
        return
    
    limit_idx=0

    for image_file, bbox_file in zip(image_files, bbox_files):
        # Load image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load bounding boxes
        bbox_path = os.path.join(bbox_dir, bbox_file)
        bboxes = np.load(bbox_path)
        
        # Draw bounding boxes
        for bbox in bboxes:
            _, x_min, y_min, x_max, y_max, _ = bbox
            cv2.rectangle(image_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        # Display the image with Matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis('off')  # Hide axis
        plt.title(f'Image: {image_file}')
        plt.show()

        # Update the limit index
        limit_idx+=1

        if (limit_idx >= limit):
            break


def transform_and_visualize(image_dir, bbox_dir, limit=10):
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

    # Get list of image files and bbox files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    bbox_files = sorted([f for f in os.listdir(bbox_dir) if f.endswith('.npy')])

    # Check if number of images and bbox files match
    if len(image_files) != len(bbox_files):
        print("The number of images and bounding box files do not match.")
        return

    limit_idx=0

    for image_file, bbox_file in zip(image_files, bbox_files):
        # Load image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load bounding boxes
        bbox_path = os.path.join(bbox_dir, bbox_file)
        bboxes = np.load(bbox_path)

        # Filter out invalid bounding boxes
        valid_bboxes = []
        for _, x_min, y_min, x_max, y_max, _ in bboxes:
            if x_max > x_min and y_max > y_min:
                valid_bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Apply transformations if there are valid bounding boxes
        if valid_bboxes:
            transformed = transform(image=image_rgb, bboxes=valid_bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            
            # Draw transformed bounding boxes
            for bbox in transformed_bboxes:
                x_min, y_min, x_max, y_max = bbox
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                cv2.rectangle(transformed_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
            # Display the image with Matplotlib
            plt.figure(figsize=(10, 10))
            plt.imshow(transformed_image)
            plt.axis('off')  # Hide axis
            plt.title(f'Image: {image_file}')
            plt.show()
        else:
            print(f"No valid bounding boxes for {image_file}")

        # Update the limit index
        limit_idx+=1

        if (limit_idx >= limit):
            break


def visualize2D(image_rgb, bboxes):
    # Draw bounding boxes
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image_rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    
    # Display the image with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.title(f'Image with Bounding Boxes')
    plt.show()

#'''
# Example usage
image_dir = r'Project\data\images'
bbox_dir = r'Project\data\bounding_boxes'
image_dir_wout_floor = r'Project\data\yolo_training\train\images'
label_dir_wout_floor = r'Project\data\yolo_training\train\labels'
limit = 3  #if you want to see the entire dataset, set the limit to a number greater than or equal to the dataset length

draw_with_bounding_boxes(image_dir, bbox_dir, limit)
draw_with_bounding_boxes(image_dir_wout_floor, label_dir_wout_floor, limit)
#transform_and_visualize(image_dir, bbox_dir, limit)
#'''
