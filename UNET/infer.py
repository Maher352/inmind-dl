# infer.py

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import load_checkpoint
from PIL import Image
import numpy as np
import cv2

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 160  # Same as training height
IMAGE_WIDTH = 240  # Same as training width
CHECKPOINT_PATH = "my_checkpoint.pth.tar"

def transform_image(image_path):
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    image = np.array(Image.open(image_path).convert("RGB"))
    image = transform(image=image)["image"]
    return image

def predict(image_path, model):
    model.eval()
    image = transform_image(image_path).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model(image)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    return prediction.cpu().squeeze().numpy()

def save_prediction(prediction, output_path):
    prediction_image = (prediction * 255).astype(np.uint8)
    cv2.imwrite(output_path, prediction_image)

def main(image_path, output_path):
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    prediction = predict(image_path, model)
    save_prediction(prediction, output_path)

if __name__ == "__main__":
    image_path = "test2.jpeg"  # Replace with your image path
    output_path = "mask.png"  # Replace with desired output path
    main(image_path, output_path)
    print(f"Prediction saved to {output_path}")
