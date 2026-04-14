import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse

IMG_SIZE = 128
MODEL_PATH = 'best_autoencoder.keras'

def preprocess_image(image_path):
    """Loads the image and prepares it for the model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
        
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    
    img_expanded = np.expand_dims(img, axis=(0, -1))
    return img, img_expanded

def show_results(original, reconstructed):
    """Displays the original, reconstruction, and difference map."""
    diff = np.abs(original - reconstructed.squeeze())
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original / Input")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Reconstruction (Autoencoder)")
    plt.imshow(reconstructed.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Difference Map (Anomalies)")
    plt.imshow(diff, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Testing the saved autoencoder on a single image.")
    parser.add_argument("image_path", help="Path to the image you want to check.")
    args = parser.parse_args()

    print(f"Loading model from: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Run train_autoencoder.py first.")
        return

    model = keras.models.load_model(MODEL_PATH, compile=False)

    # 2. Loading the image
    print(f"Processing image: {args.image_path}...")
    try:
        orig_img, input_img = preprocess_image(args.image_path)
    except Exception as e:
        print(e)
        return

    # 3. Prediction
    print("Performing prediction...")
    reconstructed_img = model.predict(input_img)

    # 4. Displaying results
    show_results(orig_img, reconstructed_img)

if __name__ == "__main__":
    main()