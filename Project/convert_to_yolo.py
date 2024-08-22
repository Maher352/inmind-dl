# convert_to_yolo

import os
import numpy as np
import cv2

def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """
    Converts a bounding box from (x_min, y_min, x_max, y_max) format to YOLO format (x_center, y_center, width, height).
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def save_bboxes_as_yolo_format(image_dir, bbox_dir, label_map):
    """
    Converts bounding boxes from numpy arrays to YOLO format and saves them as .txt files.
    """
    for bbox_file in os.listdir(bbox_dir):
        if bbox_file.endswith('.npy'):
            # Load bounding boxes
            bboxes = np.load(os.path.join(bbox_dir, bbox_file), allow_pickle=True)
            bboxes = np.array([list(item) for item in bboxes], dtype=np.float32)

            mask = bboxes[:, 0] != 3
            filtered_bboxes = bboxes[mask]

            valid_bboxes = []
            for bbox_semantic_id, x_min, y_min, x_max, y_max, _ in filtered_bboxes:
                if x_max > x_min and y_max > y_min:
                    valid_bboxes.append((int(bbox_semantic_id), int(x_min), int(y_min), int(x_max), int(y_max)))

            # Extract the corresponding image size
            img_file = bbox_file.replace('bounding_box_2d_tight_', 'rgb_').replace('.npy', '.png')
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            # Prepare the corresponding .txt file
            txt_file_path = os.path.join(bbox_dir, img_file.replace('.png', '.txt'))
            with open(txt_file_path, 'w') as f:
                for bbox in valid_bboxes:
                    class_id, x_min, y_min, x_max, y_max = bbox
                    # Convert bounding box to YOLO format
                    x_center, y_center, width, height = convert_bbox_to_yolo_format((x_min, y_min, x_max, y_max), img_width, img_height)
                    # Write to .txt file in YOLO format
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Example usage
train_image_dir = r'Project/data/yolo_training/train/images'
train_bbox_dir = r'Project/data/yolo_training/train/labels'

validate_image_dir = r'Project/data/yolo_training/validate/images'
validate_bbox_dir = r'Project/data/yolo_training/validate/labels'

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

save_bboxes_as_yolo_format(train_image_dir, train_bbox_dir, label_map)
print(f"Training data saved in yolo format successfully.")

save_bboxes_as_yolo_format(validate_image_dir, validate_bbox_dir, label_map)
print(f"Validation data saved in yolo format successfully.")