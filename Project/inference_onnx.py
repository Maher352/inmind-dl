# inference_onnx.py

'''
This file is used for infering the yolov5 model using onnx
even though it is functional the bounding boxes are missplaced, it cannot be used yet
!!!Debugging is needed, FILE NOT READY!!!

the export worked... the onnx file can be loaded from a separate folder local to 'Project'
this feature was reversed since the file is not ready anyways
'''

import time
import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the ONNX model
onnx_model_path = r'yolov5\runs\train\exp7\weights\best.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Load and preprocess the image
image_path = r'Project\data\images\rgb_0045.png'
image = cv2.imread(image_path)
original_h, original_w = image.shape[:2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (512, 512))
image_input = image_resized.astype(np.float32) / 255.0
image_input = np.transpose(image_input, (2, 0, 1))  # Change to (C, H, W)
image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension

# Perform inference
start_time = time.time()
ort_inputs = {ort_session.get_inputs()[0].name: image_input}
ort_outs = ort_session.run(None, ort_inputs)
predictions = ort_outs[0]
onnx_time = time.time() - start_time

print(f"ONNX Inference Time: {onnx_time:.6f} seconds")

# Post-process predictions for ONNX
boxes = predictions[..., :4]
scores = predictions[..., 4]
labels = predictions[..., 5]

# Rescale bounding boxes to original image size
boxes[..., 0] *= original_w / 512  # x1
boxes[..., 1] *= original_h / 512  # y1
boxes[..., 2] *= original_w / 512  # x2
boxes[..., 3] *= original_h / 512  # y2

# Draw bounding boxes on the image
for i in range(boxes.shape[0]):
    for j in range(boxes.shape[1]):
        score = scores[i, j]
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i, j]
            label = int(labels[i, j])
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'Class {label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
