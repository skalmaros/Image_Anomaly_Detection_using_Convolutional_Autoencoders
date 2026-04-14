import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = 128
NO_ANOMALY_DIR = "./data/NoAnomaly"
ANOMALY_DIR = "./data/Anomaly"
EPOCHS = 150
BATCH_SIZE = 128

def load_images(folder, limit=1000, img_size=IMG_SIZE):
    images = []
    if not os.path.exists(folder):
        print(f"Ostrzeżenie: Folder {folder} nie istnieje!")
        return np.array([])

    for i, filename in enumerate(os.listdir(folder)):
        if i >= limit:
            break
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32') / 255.0
            images.append(img)
            
    images_np = np.array(images)
    if len(images_np) > 0:
        images_np = np.reshape(images_np, (len(images_np), img_size, img_size, 1))
    return images_np

def display(array1, array2, n=10):
    n = min(n, len(array1))
    indices = np.arange(n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        ax.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
        ax.axis('off')
    plt.show()

def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy_images, 0.0, 1.0)

def mask_center(images, mask_size=14):
    masked = images.copy()
    h, w = images.shape[1:3]
    start = h // 2 - mask_size // 2
    end = start + mask_size
    masked[:, start:end, start:end, :] = 0.0
    return masked


def get_perceptual_loss_function():

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    vgg.trainable = False
    feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)

    def perceptual_loss(y_true, y_pred):
        y_true_rgb = tf.image.grayscale_to_rgb(y_true)
        y_pred_rgb = tf.image.grayscale_to_rgb(y_pred)
        true_features = feature_extractor(y_true_rgb)
        pred_features = feature_extractor(y_pred_rgb)
        return tf.reduce_mean(tf.square(true_features - pred_features))
    
    return perceptual_loss

def build_autoencoder_perceptual(loss_fn):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs, outputs, name="Perceptual_Autoencoder")
    model.compile(optimizer="adam", loss=loss_fn)
    return model


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample_image, interval=5):
        super().__init__()
        self.sample_image = sample_image 
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            reconstructed = self.model.predict(self.sample_image, verbose=0)[0]
            original = self.sample_image[0]
            diff = np.abs(original - reconstructed)
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.title("Oryginał")
            plt.imshow(original.squeeze(), cmap='gray')
            plt.subplot(1,3,2)
            plt.title("Rekonstrukcja")
            plt.imshow(reconstructed.squeeze(), cmap='gray')
            plt.subplot(1,3,3)
            plt.title("Mapa różnic")
            plt.imshow(diff.squeeze(), cmap='hot')
            plt.colorbar()
            plt.suptitle(f"Epoka: {epoch + 1}")
            plt.show()


def main():
    print("Data processing...")
    image_data = load_images(NO_ANOMALY_DIR)
    test_anomaly_data = load_images(ANOMALY_DIR)

    if len(image_data) == 0:
        print("No trianing data.")
        return

    train_data, test_data = train_test_split(image_data, test_size=0.2, random_state=42)
    
    print("Inpainting...")
    x_train_masked = mask_center(train_data)
    x_test_masked = mask_center(test_data)

    print("Creating model...")
    perceptual_loss = get_perceptual_loss_function()
    autoencoder = build_autoencoder_perceptual(perceptual_loss)
    autoencoder.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    
    checkpoint = ModelCheckpoint(
        filepath='best_autoencoder.keras',
        monitor='val_loss',                
        save_best_only=True,               
        verbose=1
    )
    
    sample_for_cb = test_data[0:1] 
    display_cb = DisplayCallback(sample_image=sample_for_cb, interval=10)
    print("Start of training...")
    autoencoder.fit(
        x=x_train_masked,
        y=train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test_masked, test_data),
        callbacks=[early_stop, checkpoint, display_cb]
    )

    print("Evaluation of the model on test data...")
    predictions = autoencoder.predict(test_data[:10])
    display(test_data[:10], predictions)

if __name__ == "__main__":
    main()