# visualize_netron.py

'''
This file is used to visualize the Network using Netron
'''

import netron

# Path to your ONNX model
onnx_model_path = 'yolov5/runs/train/exp7/weights/best.onnx'

# Visualize the model using Netron
netron.start(onnx_model_path)