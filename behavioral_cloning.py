# ============================================================================
# A. Setup and Constants (Driver Script)
# ============================================================================

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
INPUT_SHAPE = (66, 200, 3)  # Height, Width, Channels (after cropping/resizing)
LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.2

# Data augmentation parameters
STEERING_CORRECTION = 0.25  # Correction for left/right camera images
FLIP_PROBABILITY = 0.5
BRIGHTNESS_RANGE = (0.4, 1.2)
SHIFT_RANGE = 20  # pixels
SHADOW_PROBABILITY = 0.5
BLUR_PROBABILITY = 0.3


# ============================================================================
# B. Data Augmentation and Preprocessing (The Generator)
# ============================================================================

def load_driving_log(log_file_path):
    """
    Load the driving log CSV file containing image paths and steering angles.

    Args:
        log_file_path: Path to the driving_log.csv file

    Returns:
        samples: List of samples containing [center, left, right, steering, throttle, brake, speed]
    """
    samples = []
    with open(log_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header if present
        for line in reader:
            samples.append(line)
    return samples


def crop_and_resize(image):
    # Crop: remove top 60 pixels (sky) and bottom 25 pixels (hood)
    cropped = image[60:135, :, :]
    # Resize to INPUT_SHAPE
    resized = cv2.resize(cropped, (INPUT_SHAPE[1], INPUT_SHAPE[0]), interpolation=cv2.INTER_AREA)
    return resized


def flip_image(image, steering_angle):
    """
    Horizontally flip the image and negate the steering angle.
    This eliminates left/right turn bias in the training data.

    Args:
        image: Input image
        steering_angle: Original steering angle

    Returns:
        Flipped image and negated steering angle
    """
    flipped_image = cv2.flip(image, 1)
    flipped_angle = -steering_angle
    return flipped_image, flipped_angle


def random_shift(image, steering_angle, shift_range=SHIFT_RANGE):
    """
    Apply random horizontal and vertical shifts to simulate off-center driving.
    Adjust steering angle proportionally to horizontal shift.

    Args:
        image: Input image
        steering_angle: Original steering angle
        shift_range: Maximum shift in pixels

    Returns:
        Shifted image and adjusted steering angle
    """
    rows, cols, _ = image.shape

    # Random horizontal and vertical shift
    shift_x = np.random.randint(-shift_range, shift_range + 1)
    shift_y = np.random.randint(-shift_range // 2, shift_range // 2 + 1)

    # Adjust steering angle based on horizontal shift
    # 0.004 steering angle per pixel shift is a reasonable correction
    adjusted_angle = steering_angle + shift_x * 0.004

    # Create translation matrix
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

    return shifted_image, adjusted_angle


def random_brightness(image):
    """
    Adjust image brightness randomly to generalize across different
    lighting/weather conditions.

    Args:
        image: Input image in RGB format

    Returns:
        Image with adjusted brightness
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Random brightness factor
    brightness_factor = np.random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])

    # Adjust V channel (brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)

    # Convert back to RGB
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return adjusted_image


def random_shadow(image):
    """
    Apply random shadows to simulate varying road conditions.

    Args:
        image: Input image in RGB format

    Returns:
        Image with random shadow
    """
    rows, cols, _ = image.shape

    # Randomly choose shadow vertices
    x1, y1 = cols * np.random.rand(), 0
    x2, y2 = cols * np.random.rand(), rows

    # Create shadow mask
    mask = np.zeros((rows, cols), dtype=np.uint8)
    vertices = np.array([[x1, y1], [x2, y2], [cols, rows], [cols, 0]], dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 255)

    # Apply shadow (darken the masked region)
    shadow_intensity = np.random.uniform(0.3, 0.7)

    # Convert to HSV and darken V channel in shadow region
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.where(mask == 255,
                             hsv[:, :, 2] * shadow_intensity,
                             hsv[:, :, 2]).astype(np.uint8)

    shadowed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return shadowed_image


def random_blur(image):
    """
    Apply random Gaussian blur to simulate camera limitations.

    Args:
        image: Input image

    Returns:
        Blurred image
    """
    kernel_size = np.random.choice([3, 5])
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


def augment_image(image, steering_angle, is_training=True):
    """
    Apply comprehensive augmentation pipeline to a single image.

    Args:
        image: Input image
        steering_angle: Original steering angle
        is_training: Whether to apply augmentation (True for training, False for validation)

    Returns:
        Augmented image and adjusted steering angle
    """
    if not is_training:
        return image, steering_angle

    # Random flip
    if np.random.rand() < FLIP_PROBABILITY:
        image, steering_angle = flip_image(image, steering_angle)

    # Random shift
    if np.random.rand() < 0.5:
        image, steering_angle = random_shift(image, steering_angle)

    # Random brightness
    if np.random.rand() < 0.8:
        image = random_brightness(image)

    # Random shadow
    if np.random.rand() < SHADOW_PROBABILITY:
        image = random_shadow(image)

    # Random blur
    if np.random.rand() < BLUR_PROBABILITY:
        image = random_blur(image)

    return image, steering_angle


def image_data_generator(samples, image_dir, batch_size=BATCH_SIZE, is_training=True):
    """
    Generator function that yields batches of augmented and preprocessed images.

    Implements multi-camera correction by using left, center, and right camera images
    with appropriate steering angle corrections.

    Args:
        samples: List of samples from driving log
        image_dir: Base directory containing IMG folder
        batch_size: Number of samples per batch
        is_training: Whether to apply augmentation

    Yields:
        Batches of (images, steering_angles)
    """
    num_samples = len(samples)

    while True:  # Loop forever for Keras fit_generator
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []

            for batch_sample in batch_samples:
                # Extract paths and steering angle
                center_path = batch_sample[0].strip()
                left_path = batch_sample[1].strip()
                right_path = batch_sample[2].strip()
                steering_center = float(batch_sample[3])

                # Use all three camera images with steering correction
                camera_images = []
                camera_angles = []

                # Center camera
                center_name = os.path.join(image_dir, 'IMG', os.path.basename(center_path))
                if os.path.exists(center_name):
                    center_image = cv2.imread(center_name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    camera_images.append(center_image)
                    camera_angles.append(steering_center)

                # Left camera (steer right to recover)
                left_name = os.path.join(image_dir, 'IMG', os.path.basename(left_path))
                if os.path.exists(left_name):
                    left_image = cv2.imread(left_name)
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    camera_images.append(left_image)
                    camera_angles.append(steering_center + STEERING_CORRECTION)

                # Right camera (steer left to recover)
                right_name = os.path.join(image_dir, 'IMG', os.path.basename(right_path))
                if os.path.exists(right_name):
                    right_image = cv2.imread(right_name)
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    camera_images.append(right_image)
                    camera_angles.append(steering_center - STEERING_CORRECTION)

                # Process each camera image
                for img, angle in zip(camera_images, camera_angles):
                    # Crop and resize
                    processed_img = crop_and_resize(img)

                    # Apply augmentation
                    augmented_img, augmented_angle = augment_image(
                        processed_img, angle, is_training=is_training
                    )

                    images.append(augmented_img)
                    steering_angles.append(augmented_angle)

            # Convert to numpy arrays
            X_batch = np.array(images)
            y_batch = np.array(steering_angles)

            yield shuffle(X_batch, y_batch)


# ============================================================================
# C. Model Architecture (NVIDIA-Inspired CNN)
# ============================================================================

def create_nvidia_model():
    """
    Create the NVIDIA-inspired CNN model for end-to-end autonomous driving.

    Architecture:
    - Normalization layer
    - 5 Convolutional layers with ELU activation
    - Dropout for regularization
    - Flatten layer
    - 3 Fully connected layers with ELU activation
    - Output layer (steering angle prediction)

    Returns:
        Keras Sequential model
    """
    model = Sequential([
        # Normalization layer: normalize pixel values to [-1, 1]
        Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE),

        # Convolutional Layer 1: 24 filters, 5x5 kernel, 2x2 stride
        Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='valid'),

        # Convolutional Layer 2: 36 filters, 5x5 kernel, 2x2 stride
        Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='valid'),

        # Convolutional Layer 3: 48 filters, 5x5 kernel, 2x2 stride
        Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu', padding='valid'),

        # Convolutional Layer 4: 64 filters, 3x3 kernel
        Conv2D(64, kernel_size=(3, 3), activation='elu', padding='valid'),

        # Convolutional Layer 5: 64 filters, 3x3 kernel
        Conv2D(64, kernel_size=(3, 3), activation='elu', padding='valid'),

        # Dropout layer to prevent overfitting
        Dropout(0.5),

        # Flatten layer
        Flatten(),

        # Fully connected layer 1: 100 neurons
        Dense(100, activation='elu'),

        # Fully connected layer 2: 50 neurons
        Dense(50, activation='elu'),

        # Fully connected layer 3: 10 neurons
        Dense(10, activation='elu'),

        # Output layer: 1 neuron (steering angle)
        Dense(1)
    ])

    return model


# ============================================================================
# D. Training and Saving
# ============================================================================

def train_model(log_file_path, image_dir, model_save_path='model.h5'):
    """
    Complete training pipeline for the behavioral cloning model.

    Args:
        log_file_path: Path to driving_log.csv
        image_dir: Base directory containing IMG folder
        model_save_path: Path to save the best model
    """
    print("=" * 80)
    print("BEHAVIORAL CLONING - AUTONOMOUS DRIVING")
    print("=" * 80)

    # Load driving log
    print("\n[1/6] Loading driving log...")
    samples = load_driving_log(log_file_path)
    print(f"Total samples loaded: {len(samples)}")

    # Split data into training and validation sets
    print("\n[2/6] Splitting data into training and validation sets...")
    train_samples, validation_samples = train_test_split(
        samples, test_size=VALIDATION_RATIO, random_state=42
    )
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")

    # Create data generators
    print("\n[3/6] Creating data generators...")
    train_generator = image_data_generator(
        train_samples, image_dir, batch_size=BATCH_SIZE, is_training=True
    )
    validation_generator = image_data_generator(
        validation_samples, image_dir, batch_size=BATCH_SIZE, is_training=False
    )

    # Calculate steps per epoch (multiply by 3 for multi-camera)
    train_steps = (len(train_samples) * 3) // BATCH_SIZE
    validation_steps = (len(validation_samples) * 3) // BATCH_SIZE

    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {validation_steps}")

    # Create model
    print("\n[4/6] Building NVIDIA-inspired CNN model...")
    model = create_nvidia_model()

    # Display model architecture
    model.summary()

    # Compile model with Adam optimizer
    print("\n[5/6] Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )

    # Setup ModelCheckpoint callback to save best model
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    # Train the model
    print("\n[6/6] Training model...")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 80)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best model saved to: {model_save_path}")
    print("=" * 80)

    # Plot training history
    plot_training_history(history)

    return model, history


def plot_training_history(history):
    """
    Plot training and validation loss over epochs.

    Args:
        history: Keras History object from model.fit()
    """
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: training_history.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Configuration
    LOG_FILE = 'driving_log.csv'  # Path to driving log
    IMAGE_DIR = '.'  # Base directory containing IMG folder
    MODEL_SAVE_PATH = 'model_best.h5'  # Path to save best model

    # Check if files exist
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found!")
        print("Please ensure driving_log.csv is in the working directory.")
        exit(1)

    if not os.path.exists(os.path.join(IMAGE_DIR, 'IMG')):
        print(f"Error: IMG folder not found in {IMAGE_DIR}!")
        print("Please ensure the IMG folder with training images is available.")
        exit(1)

    # Train model
    model, history = train_model(LOG_FILE, IMAGE_DIR, MODEL_SAVE_PATH)

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Load the model: model = keras.models.load_model('model_best.h5')")
    print("2. Use drive.py to test the model in Udacity simulator")
    print("3. Monitor performance on both Track 1 and Track 2")
    print("=" * 80)
