import argparse
import base64
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from datetime import datetime

from tensorflow import keras

# Initialize Flask and SocketIO
sio = socketio.Server()
app = Flask(__name__)

# Global variables
model = None
prev_image_array = None
target_speed = 9
speed_limit = 15


def crop_and_resize(image):
    # Crop: remove top 60 pixels (sky) and bottom 25 pixels (hood)
    cropped = image[60:135, :, :]
    # Resize to (66, 200)
    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)
    return resized


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Current steering angle from simulator
        steering_angle = float(data["steering_angle"])
        # Current throttle
        throttle = float(data["throttle"])
        # Current speed
        speed = float(data["speed"])
        # Current brake
        brake = float(data["brake"])

        # Decode image from base64
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image_array = np.asarray(image)

        # Preprocess image (same as training)
        preprocessed = crop_and_resize(image_array)

        # Normalize to [-1, 1] (model expects this)
        normalized = preprocessed / 127.5 - 1.0

        # Predict steering angle
        # Model expects shape (batch_size, height, width, channels)
        predicted_angle = float(model.predict(normalized[None, :, :, :], verbose=0))

        # Simple speed controller
        # If speed is below target, accelerate; otherwise, coast
        if speed < target_speed:
            throttle_value = 0.3
        elif speed < target_speed + 2:
            throttle_value = 0.1
        else:
            throttle_value = 0.0

        # Apply speed limit for safety
        if speed > speed_limit:
            throttle_value = 0.0

        # Send control commands to simulator
        send_control(predicted_angle, throttle_value)

        # Optional: Save frames for debugging
        # save_frame(image_array, predicted_angle, speed)

        # Log telemetry
        print(f"Speed: {speed:6.2f} mph | Steering: {predicted_angle:7.4f} | Throttle: {throttle_value:4.2f}")

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    """Handle client connection."""
    print("Connected to simulator!")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True
    )


def save_frame(image, steering_angle, speed):
    """
    Optional: Save frames for debugging or video creation.

    Args:
        image: Image array
        steering_angle: Predicted steering angle
        speed: Current speed
    """
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    filename = f'run_images/{timestamp}.jpg'

    # Draw telemetry on image
    annotated = image.copy()
    cv2.putText(annotated, f'Angle: {steering_angle:.3f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f'Speed: {speed:.2f} mph', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(filename, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autonomous Driving with Behavioral Cloning')
    parser.add_argument(
        'model',
        type=str,
        help='Path to trained model (e.g., model_best.h5)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=9,
        help='Target speed for the car (default: 9 mph)'
    )
    args = parser.parse_args()

    # Load the trained model
    print("=" * 80)
    print("AUTONOMOUS DRIVING MODE")
    print("=" * 80)
    print(f"\nLoading model: {args.model}")

    model = keras.models.load_model(args.model)
    print("Model loaded successfully!")

    # Set target speed
    target_speed = args.speed
    print(f"Target speed: {target_speed} mph")
    print(f"Speed limit: {speed_limit} mph")

    print("\n" + "-" * 80)
    print("Starting server...")
    print("Please start the Udacity simulator and select AUTONOMOUS MODE")
    print("-" * 80 + "\n")

    # Wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
