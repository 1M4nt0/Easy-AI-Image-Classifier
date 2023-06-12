import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.models import resnet50
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models.resnet import ResNet50_Weights
import sqlite3

# Create a SQLite connection and cursor
conn = sqlite3.connect("embeddings.db")
cursor = conn.cursor()

def create_table():
    # Create a table to store the embeddings if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings
                      (image_path TEXT PRIMARY KEY, embedding BLOB)''')
    conn.commit()

def save_embedding(image_path, embedding):
    # Save the embedding into the database
    cursor.execute("INSERT OR REPLACE INTO embeddings (image_path, embedding) VALUES (?, ?)", (image_path, embedding.tobytes()))
    conn.commit()

def load_embedding(image_path):
    # Load the embedding from the database
    cursor.execute("SELECT embedding FROM embeddings WHERE image_path=?", (image_path,))
    result = cursor.fetchone()
    if result is not None:
        return np.frombuffer(result[0], dtype=np.float32)
    else:
        return None

def compute_similarity(embedding1, embedding2):
    # Reshape the embeddings if necessary
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Compute cosine similarity between the embeddings
    similarity = cosine_similarity(embedding1, embedding2)[0, 0]
    return similarity

def load_pretrained_model():
    # Load the pre-trained ResNet-50 model
    model = model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Remove the last fully connected layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    # Set the model to evaluation mode
    model.eval()
    return model

def compute_image_embeddings(model, image):
    image_tensor = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        embeddings = model(image_tensor)
    return embeddings.squeeze().numpy()

def compute_similarity_with_folder(image, folder, model):
    image_files = os.listdir(folder)
    similarities = []

    for image_name in image_files:
        image_path = os.path.join(folder, image_name)
        if os.path.isfile(image_path):
            test_image = Image.open(image_path).convert('RGB')
            embedding1 = compute_image_embeddings(model, test_image)
            embedding2 = compute_image_embeddings(model, image)
            similarity = compute_similarity(embedding1, embedding2)
            similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.0

def find_best_matching_folder(image, group_folder, model):
    image_files = os.listdir(group_folder)
    best_similarity = -1
    best_folder = None

    for folder_name in image_files:
        folder_path = os.path.join(group_folder, folder_name)
        if os.path.isdir(folder_path):
            test_folder_path = os.path.join(folder_path, "test")
            if os.path.isdir(test_folder_path):
                # Compute the mean similarity score with all images in the test folder
                test_images = os.listdir(test_folder_path)
                mean_similarity = 0
                for test_image_name in test_images:
                    test_image_path = os.path.join(test_folder_path, test_image_name)
                    if os.path.isfile(test_image_path):
                        # Load the test image and compute embeddings
                        embedding1 = load_embedding(test_image_path)
                        if embedding1 is None:
                            test_image = Image.open(test_image_path).convert('RGB')
                            embedding1 = compute_image_embeddings(model, test_image)
                            save_embedding(test_image_path, embedding1)

                        embedding2 = compute_image_embeddings(model, image)
                        similarity = compute_similarity(embedding1, embedding2)
                        mean_similarity += similarity

                mean_similarity /= len(test_images)

                print(f"Mean similarity score with images in {test_folder_path}: {mean_similarity}")

                # Update the best similarity and folder
                if mean_similarity > best_similarity:
                    best_similarity = mean_similarity
                    best_folder = folder_path

    return best_folder, best_similarity

def move_image_to_best_folder(images_folder, group_folder):
    # Load the pre-trained model
    model = load_pretrained_model()

    # Search for images recursively in the "images" folder
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            print(f"Evaluating {file}")
            image_path = os.path.join(root, file)
            if os.path.isfile(image_path):
                image = Image.open(image_path).convert('RGB')
                best_folder, best_similarity = find_best_matching_folder(image, group_folder, model)
                if best_folder:
                    if best_similarity >= 0.9:
                        shutil.move(image_path, best_folder)
                    else:
                        altro_folder = os.path.join(group_folder, "altro")
                        if not os.path.exists(altro_folder):
                            os.makedirs(altro_folder)
                        shutil.move(image_path, altro_folder)


# Create the table to store embeddings
create_table()

# Specify the paths to the folders
images_folder = "images"
group_folder = "groups"

# Call the function to move the images
move_image_to_best_folder(images_folder, group_folder)

# Close the database connection
conn.close()