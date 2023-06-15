# Image Classification with ResNet-50

This script allows you to perform image classification using the ResNet-50 model. It consists of two main parts: training the model and performing inference on new images.

## Training

The training process involves the following steps:

1. Conversion of PNG images to RGBA format.
2. Random splitting of images into training and validation sets.
3. Loading the pre-trained ResNet-50 model.
4. Training the model on the training dataset.
5. Evaluating the model on the validation dataset.
6. Saving the trained model.

### Requirements

Make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- scikit-learn

### Usage

To train the model, follow these steps:

1. Prepare your dataset by placing the images in the `data` folder. If any PNG images are present, they will be automatically converted to RGBA format.
2. Adjust the `num_categories` variable in the script to match the number of categories in your dataset.
3. Run the script.

After training, the model will be saved as `trained_model.pth` in the current directory.

## Inference

The inference process uses a pre-trained ResNet-50 model to classify new images. Follow these steps:

### Requirements

Ensure that the following dependencies are installed:

- Python 3.x
- PyTorch
- torchvision

### Usage

To perform inference on new images, follow these steps:

1. Prepare the input images by placing them in the `input` folder.
2. Make sure you have the trained model weights saved as `trained_model.pth` in the current directory.
3. Adjust the `num_categories` variable in the script to match the number of categories in your dataset.
4. Run the script.

The script will classify each input image using the trained model and move it to the corresponding category folder in the `output` directory.

Feel free to customize the script according to your specific requirements.
