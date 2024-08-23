# inference_torch.py

# !!!THIS IS DEPENDENT ON THE LOCAL TRAINNING = CAN ONLY RUN ON THE DEVICE WHERE THE TRAINNING WAS DONE!!!

'''
This file is used to infere the trained model using the pyTorch library
'''

import time
import cv2
import torch
from PIL import Image

model_path = 'yolov5/runs/train/exp7/weights/best.pt'
img_path = 'Project/data/yolo_training/validate/images/rgb_0045.png'

# Model
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local') 

img = Image.open(img_path)  # PIL image

img = img.resize((512,512))

start_time = time.time()
# Inference
results = model(img, size=512)  # includes NMS

torch_time = time.time() - start_time

# Results
results.print()  # print results to screen
results.show()  # display results
results.save()  # save as results1.jpg, results2.jpg... etc.

# Data
print('\n', results.xyxy[0])  # print img1 predictions