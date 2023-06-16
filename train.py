import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
from utils import count_folders, remove_folder

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

model_name = input("Define the name for your model: ")

# Define the percentage of images to allocate for training (e.g., 80%)
train_percentage = float(input("Define the percentage of images to allocate for training (e.g., 0.8): "))

batch_size = int(input("Define the batch size for training your model (e.g., 16): "))

num_epochs = int(input("Define the number of training epochs: "))

num_categories = count_folders("data")  # Change this to match your dataset

remove_folder("training")
remove_folder("validation")
remove_folder("output")


# Define the path to the original "data" folder
original_folder = './data'

print("\n\nConverting PNGs images to RGBA")
# Recursively search for PNG images and convert them to RGBA
for root, dirs, files in os.walk(original_folder):
    for file in files:
        if file.endswith('.png'):
            image_path = os.path.join(root, file)
            im = Image.open(image_path)
            if im.format == 'PNG' and im.mode != 'RGBA':
                im = im.convert('RGBA')
                im.save(image_path)

print("Conversion completed!\n\n")

print(f"Categories found: {num_categories}\n\n")

# Define the path to the new "training" and "validation" folders
training_folder = './training'
validation_folder = './validation'

# Create the "training" and "validation" folders if they don't exist
os.makedirs(training_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Randomly split images into training and validation folders
for root, dirs, files in os.walk(original_folder):
    # Get the relative path from the original folder
    relative_path = os.path.relpath(root, original_folder)

    # Create the corresponding folders in the "training" and "validation" directories
    training_dir = os.path.join(training_folder, relative_path)
    validation_dir = os.path.join(validation_folder, relative_path)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Randomly shuffle the list of files
    random.shuffle(files)

    # Split the files based on the train_percentage
    train_size = int(len(files) * train_percentage)
    train_files = files[:train_size]
    validation_files = files[train_size:]

    # Move the files to the "training" and "validation" folders
    for file in train_files:
        src = os.path.join(root, file)
        dst = os.path.join(training_dir, file)
        shutil.copy(src, dst)

    for file in validation_files:
        src = os.path.join(root, file)
        dst = os.path.join(validation_dir, file)
        shutil.copy(src, dst)


# Load pre-trained ResNet-50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_categories)

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the ImageFolder dataset for training
training_dataset = ImageFolder(training_folder, transform=transform)

# Create a data loader for the training dataset
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

# Create the ImageFolder dataset for validation
validation_dataset = ImageFolder(validation_folder, transform=transform)

# Create a data loader for the validation dataset
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Training...")
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    # Validation loop
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        val_total_labels = []
        val_total_predictions = []

        for val_images, val_labels in validation_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            _, val_predicted = torch.max(val_outputs.data, 1)

            val_total_labels.extend(val_labels.cpu().numpy())
            val_total_predictions.extend(val_predicted.cpu().numpy())

    val_accuracy = round(accuracy_score(val_total_labels, val_total_predictions),4)
    val_loss = round(running_loss / len(training_loader), 4)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the trained model
torch.save(model.state_dict(), f'models/{model_name}_e{num_epochs}_c{num_categories}.pth')
