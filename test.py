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
from torchvision.models.resnet import ResNet50_Weights
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the path to the original "data" folder
original_folder = './data'

# Define the path to the new "training" and "validation" folders
training_folder = './training'
validation_folder = './validation'

# Create the "training" and "validation" folders if they don't exist
os.makedirs(training_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Define the percentage of images to allocate for training (e.g., 80%)
train_percentage = 0.8

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

# Convert palette images to RGBA format
def convert_palette_to_rgba(image):
    if isinstance(image, int):
        return image  # Skip non-image targets
    if image.mode == 'P' and 'transparency' in image.info:
        try:
            image = image.convert('RGBA')
        except Exception:
            pass
    return image


# Modify the last layer for your own categories
num_categories = 11  # Change this to match your dataset
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_categories)

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the ImageFolder dataset for training
training_dataset = ImageFolder(training_folder, transform=transform, target_transform=convert_palette_to_rgba)

# Create a data loader for the training dataset
batch_size = 32
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

# Create the ImageFolder dataset for validation
validation_dataset = ImageFolder(validation_folder, transform=transform, target_transform=convert_palette_to_rgba)

# Create a data loader for the validation dataset
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 20
device = torch.device("cpu")
model.to(device)

# Initialize variables to store metrics
total_labels = []
total_predictions = []

print("Start training...")
for epoch in range(num_epochs):
    running_loss = 0.0
    total_labels_epoch = []
    total_predictions_epoch = []

    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Convert tensor predictions and labels to numpy arrays
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()

        # Append predictions and labels for the epoch
        total_labels_epoch.extend(labels)
        total_predictions_epoch.extend(predicted)

    # Calculate metrics for the training epoch
    accuracy = round(accuracy_score(total_labels_epoch, total_predictions_epoch), 4)
    precision = round(precision_score(total_labels_epoch, total_predictions_epoch, average='macro', zero_division=1), 4)
    recall = round(recall_score(total_labels_epoch, total_predictions_epoch, average='macro', zero_division=1), 4)
    f1 = round(f1_score(total_labels_epoch, total_predictions_epoch, average='macro', zero_division=1), 4)

    # Append metrics to the overall lists
    total_labels.extend(total_labels_epoch)
    total_predictions.extend(total_predictions_epoch)

    print(f"Training - Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(training_loader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Validation loop
    model.eval()  # Set the model to evaluation mode

    val_total_labels = []
    val_total_predictions = []

    with torch.no_grad():
        for val_images, val_labels in validation_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            _, val_predicted = torch.max(val_outputs.data, 1)

            # Convert tensor predictions and labels to numpy arrays
            val_predicted = val_predicted.cpu().numpy()
            val_labels = val_labels.cpu().numpy()

            # Append predictions and labels for validation
            val_total_labels.extend(val_labels)
            val_total_predictions.extend(val_predicted)

    # Calculate metrics for the validation set
    val_accuracy = round(accuracy_score(val_total_labels, val_total_predictions), 4)
    val_precision = round(precision_score(val_total_labels, val_total_predictions, average='macro', zero_division=1), 4)
    val_recall = round(recall_score(val_total_labels, val_total_predictions, average='macro', zero_division=1), 4)
    val_f1 = round(f1_score(val_total_labels, val_total_predictions, average='macro', zero_division=1), 4)

    print(f"Validation - Epoch {epoch + 1}/{num_epochs}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
