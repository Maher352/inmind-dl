#this python file is meant for a google colab compiler only

import albumentations as A
from google.colab.patches import cv2_imshow
import cv2

# Read the image
image = cv2.imread('/content/drive/MyDrive/In Mind/Session08_7 31/data/images/000001.jpg')

# Read the text file
with open('/content/drive/MyDrive/In Mind/Session08_7 31/data/labels/000001.txt', 'r') as file:
    lines = file.readlines()
    lines.pop()
    lines.pop()

# Transforming the text file into Bounding Boxes
bboxes = []

for line in lines:
    data = line.strip().split()
    # class x_center y_center width height
    bbox = [float(coord) for coord in data[1:]]
    bbox.append(data[0]) # these 2 lines are used to put the class in the first index

    bboxes.append(bbox)

# Display the result
print(bboxes)

# Choose augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=130, p=0.5),
    A.Blur(blur_limit=299, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    #A.RandomResizedCrop(480, 480),
    A.RGBShift(r_shift_limit=200, g_shift_limit=200, b_shift_limit=200, p=0.5),
    #A.RandomBrightness(limit=0.2, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
], bbox_params=A.BboxParams(format='yolo'))

# Apply augmentations
transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']

print(transformed_bboxes)

# Convert the bounding boxes from relative to absolute coordinates
height, width, _ = transformed_image.shape

# Draw bounding boxes on the original image
for bbox in bboxes:
    x_center, y_center, width, height, _ = bbox
    img_height, img_width = image.shape[:2]
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)
    
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

# Draw bounding boxes on the augmented image
## Tuple to list
absolute_bboxes = [[bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]] for bbox in transformed_bboxes]

for bbox in absolute_bboxes:
    x_center, y_center, width, height, _ = bbox
    xmin = int((x_center - width / 2)* img_width)
    ymin = int((y_center - height / 2)* img_height)
    xmax = int((x_center + width / 2)* img_width)
    ymax = int((y_center + height / 2)* img_height)
    cv2.rectangle(transformed_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)


# Display the original and augmented images with bounding boxes
cv2_imshow(image)

cv2_imshow(transformed_image)