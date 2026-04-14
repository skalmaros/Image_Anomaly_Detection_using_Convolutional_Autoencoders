# Image Anomaly Detection using Convolutional Autoencoders

This repository contains a deep learning project that utilizes a Convolutional Autoencoder to detect anomalies in grayscale images. It includes scripts for training the model from scratch and performing inference on new images. Orginnaly it was used for detection of foreign objects in X-ray images.

## How the Autoencoder Works

An autoencoder is a specialized type of neural network that learns to compress and then reconstruct data. It consists of two main parts:
1.  **Encoder:** Takes the input image and compresses it into a lower-dimensional representation (latent space), extracting the most important features.
2.  **Decoder:** Takes this compressed representation and attempts to reconstruct the original image as accurately as possible.

### Anomaly Detection Strategy

This project leverages the autoencoder for anomaly detection using the following logic:
* **Training on Normal Data:** The model is trained exclusively on images without anomalies (the "NoAnomaly" dataset). During training, a physical mask is applied to the center of the images. The network is forced to learn how to reconstruct (inpaint) the missing parts based on the surrounding context and its understanding of "normal" data.
* **Perceptual Loss:** Instead of simple pixel-to-pixel comparison (like MSE), the model uses a Perceptual Loss function based on a pre-trained VGG16 network. This forces the autoencoder to reconstruct high-level structural features rather than just matching pixel intensities, leading to sharper and more accurate reconstructions.
* **Detecting Anomalies:** When an image containing an anomaly is passed through the network, the autoencoder attempts to reconstruct it based on what it knows about normal images. Because it has never seen the anomaly during training, it fails to reconstruct the anomalous region accurately. 
* **Difference Map:** By calculating the absolute difference between the original input image and the autoencoder's reconstructed output, we create a heatmap. The areas with the highest differences indicate the presence of anomalies.

## Project Structure

* `train_autoencoder.py` - The main script used to prepare the data, build the model architecture, and train the network. It automatically saves the best performing model.
* `predict.py` - A command-line inference script used to test the trained model on a single image and visualize the results.
* `requirements.txt` - A list of required Python libraries.
* `best_autoencoder.keras` - The saved model weights (generated after running the training script).
* `data/` - The directory where your image datasets should be stored.

## Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.8 or higher installed.
3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt

## Using the Code Programmatically

If you prefer not to use the command line, you can easily import the core functions and models into your own Python projects.

### 1. Running Inference in Your Code

To load the trained autoencoder and perform anomaly detection on an image within a custom script, you can import the preprocessing logic from `predict.py`:

```python
import numpy as np
from tensorflow import keras
from predict import preprocess_image

# 1. Load the pre-trained model
model = keras.models.load_model('best_autoencoder.keras', compile=False)

# 2. Preprocess a new image
image_path = 'data/Anomaly/sample_image.png'
original_image, input_tensor = preprocess_image(image_path)

# 3. Generate the reconstruction
reconstructed_image = model.predict(input_tensor)

# 4. Calculate the anomaly map (difference between original and reconstruction)
anomaly_map = np.abs(original_image - reconstructed_image.squeeze())

# Now you can analyze 'anomaly_map' (e.g., apply a threshold to flag defects)
