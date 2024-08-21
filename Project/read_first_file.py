# read_first_file

import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

# Define file paths
img_file = 'Project/data/images/rgb_0000.png'
bbox_file = 'Project/data/bounding_boxes/bounding_box_2d_tight_0000.npy'
bbox_labels_file = 'Project/data/bounding_boxes_labels/bounding_box_2d_tight_labels_0000.json'
#bbox_prim_path_file = 'Project/data/bounding_boxes_prim_paths/bounding_box_2d_tight_prim_paths_0000.json'
segmentation_file = 'Project/data/semantic_segmentation/semantic_segmentation_0000.png'
segmentation_labels_file = 'Project/data/semantic_segmentation_labels/semantic_segmentation_labels_0000.json'

def read_image(file_path):
    return Image.open(file_path).convert('RGB')

def read_bounding_boxes(file_path):
    return np.load(file_path)

def read_bounding_box_labels(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

'''
def read_bounding_box_prim_paths(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
'''

def read_semantic_segmentation(file_path):
    return Image.open(file_path)

def read_semantic_segmentation_labels(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Read and display the first file of each type
if __name__ == "__main__":
    # Read image
    img = read_image(img_file)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.show()

    # Read bounding boxes
    bboxes = read_bounding_boxes(bbox_file)
    print("Bounding boxes:", bboxes)
    print("Each bbox in the boundingbox_2d_tight<NUMBER>.npy file has the following attributes: ",
          "\nbbox_semantic_id, x_min, y_min, x_max, y_max, occ_rate (You can ignore the occ_rate)")

    # Read bounding box labels
    bbox_labels = read_bounding_box_labels(bbox_labels_file)
    print("Bounding box labels:", bbox_labels)

    '''
    # Read bounding box prim paths
    bbox_prim_paths = read_bounding_box_prim_paths(bbox_prim_path_file)
    print("Bounding box prim paths:", bbox_prim_paths)
    '''

    # Read semantic segmentation
    segmentation = read_semantic_segmentation(segmentation_file)
    plt.imshow(segmentation)
    plt.title('Semantic Segmentation')
    plt.axis('off')
    plt.show()

    # Read semantic segmentation labels
    segmentation_labels = read_semantic_segmentation_labels(segmentation_labels_file)
    print("Semantic segmentation labels:", segmentation_labels)