import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

torch.cuda.set_per_process_memory_fraction(0.5)

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

# Create the ImageFolder dataset
dataset = ImageFolder('./categories', transform=transform, target_transform=convert_palette_to_rgba)

# Create a data loader for the dataset
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 20
device = torch.device("cpu")
model.to(device)

print("Start training...")

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(data_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
