print("packages.py is running")

# Check Python version
import sys
print("Python version:", sys.version)

# Check torch version
import torch
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Check the number of GPUs available
gpu_count = torch.cuda.device_count()
print("Number of GPUs Available:", gpu_count)

# Print the name of each GPU
for i in range(gpu_count):
    print(f"GPU {i} Name:", torch.cuda.get_device_name(i))

# Check cv2 version
import cv2
print("Computer vision version:", cv2.__version__)

# Check albumentations version
import albumentations
print("Albumentations version:", albumentations.__version__)

# Check open3d version
import open3d as o3d
print("Open3D version:", o3d.__version__)

# Check torchvision version
import torchvision
print("Torch vision:", torchvision.__version__)
