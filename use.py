import os
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image

model_name = input("Insert the name of the model to use (without extension): ")

# Load pre-trained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

num_categories = int(input("Insert the number of categories to classify photos (c[n] of model name): "))
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_categories)

# Load the trained model weights
model.load_state_dict(torch.load(f'models/{model_name}.pth'))

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
image_folder = 'input'
image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.webp']]

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
    print(f"Moved {image_path} => {destination_path}")