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


# Accuracy: Accuracy measures the overall correctness of the model's predictions. It is the ratio of correctly predicted samples to the total number of samples. An accuracy of 1.0 indicates that all predictions were correct, while a value close to 0.0 indicates poor performance.

# Precision: Precision is the ability of the model to correctly identify positive instances out of all instances predicted as positive. It is calculated as the ratio of true positives to the sum of true positives and false positives. Precision is a useful metric when the cost of false positives is high.

# Recall (Sensitivity or True Positive Rate): Recall measures the ability of the model to correctly identify positive instances out of all actual positive instances. It is calculated as the ratio of true positives to the sum of true positives and false negatives. Recall is particularly important when it is crucial to avoid false negatives.

# F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure between precision and recall. The F1 score combines both metrics into a single value and is useful when there is an uneven class distribution or when both false positives and false negatives need to be considered.


# Initialize variables to store metrics
total_labels = []
total_predictions = []

print("Start training...")
for epoch in range(num_epochs):
    running_loss = 0.0
    total_labels_epoch = []
    total_predictions_epoch = []

    for images, labels in data_loader:
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

    # Calculate metrics for the epoch
    accuracy = round(accuracy_score(total_labels_epoch, total_predictions_epoch), 4)
    precision = round(precision_score(total_labels_epoch, total_predictions_epoch, average='macro', zero_division=1), 4)
    recall = round(recall_score(total_labels_epoch, total_predictions_epoch, average='macro', zero_division=1), 4)
    f1 = round(f1_score(total_labels_epoch, total_predictions_epoch, average='macro', zero_division=1), 4)

    # Append metrics to the overall lists
    total_labels.extend(total_labels_epoch)
    total_predictions.extend(total_predictions_epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(data_loader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Calculate overall metrics
overall_accuracy = round(accuracy_score(total_labels, total_predictions), 4)
overall_precision = round(precision_score(total_labels, total_predictions, average='macro', zero_division=1), 4)
overall_recall = round(recall_score(total_labels, total_predictions, average='macro', zero_division=1), 4)
overall_f1 = round(f1_score(total_labels, total_predictions, average='macro', zero_division=1), 4)

print(f"Overall Metrics: Accuracy: {overall_accuracy}, Precision: {overall_precision}, Recall: {overall_recall}, F1 Score: {overall_f1}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
