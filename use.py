import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import os
import shutil
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
from torch.utils.data import Dataset

# Load pre-trained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_categories = 10  # Change this to match your dataset
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_categories)

# Load the trained model weights
model.load_state_dict(torch.load('trained_model.pth'))

# Set the model to evaluation mode
model.eval()

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the output folder if it doesn't exist
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Process each image in the "images" folder
image_folder = 'images'
image_files = os.listdir(image_folder)

for image_file in image_files:
    # Load and transform the image
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Perform the classification
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_category = predicted.item()

    # Create the category folder if it doesn't exist
    category_folder = os.path.join(output_folder, str(predicted_category))
    os.makedirs(category_folder, exist_ok=True)

    # Move the image to the corresponding category folder in the output directory
    destination_path = os.path.join(category_folder, image_file)
    shutil.move(image_path, destination_path)
    print(f"Moved {image_path} to {destination_path}")