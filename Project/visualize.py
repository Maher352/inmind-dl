# visualize.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_bounding_boxes(image_dir, bbox_dir):
    # Get list of image files and bbox files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    bbox_files = sorted([f for f in os.listdir(bbox_dir) if f.endswith('.npy')])
    
    # Check if number of images and bbox files match
    if len(image_files) != len(bbox_files):
        print("The number of images and bounding box files do not match.")
        return
    
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

# Example usage
image_dir = 'Project\data\images'
bbox_dir = r'Project\data\bounding_boxes'

draw_bounding_boxes(image_dir, bbox_dir)
