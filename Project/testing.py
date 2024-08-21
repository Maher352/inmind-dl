# testing.py

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BMWDataset

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set the paths to your data directories
json_dir = "Project/data/bounding_boxes_labels"
img_dir = "Project/data/images"

# Create an instance of BMWDataset
dataset = BMWDataset(json_dir=json_dir, img_dir=img_dir, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader to check the outputs
for images, labels in dataloader:
    print(images.size(), labels)
    break  # Just checking the first batch
