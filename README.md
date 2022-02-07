# Deep Learning Model for Simulating Self Driving Car

**A Behavioral Cloning Implementation using NVIDIA's CNN Architecture**

![System Architecture](docs/system_architecture.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Network Architecture](#network-architecture)
- [Installation](#installation)
- [Dataset Collection](#dataset-collection)
- [Data Augmentation & Preprocessing](#data-augmentation--preprocessing)
- [Training the Model](#training-the-model)
- [Testing in Autonomous Mode](#testing-in-autonomous-mode)
- [Results](#results)
- [Project Structure](#project-structure)
- [References](#references)

---

## 🎯 Overview

This project implements an **end-to-end deep learning approach** for autonomous vehicle control using **Behavioral Cloning** technique in the Udacity Car Simulator environment. The convolutional neural network (CNN) learns to predict steering angles, throttle, and brakes by mimicking human driving behavior.

### Key Objectives

- Train a CNN model to autonomously control a virtual car's steering angle, throttle, and brakes
- Implement NVIDIA's proven end-to-end learning architecture
- Generalize driving behavior across different tracks using data augmentation
- Achieve autonomous driving on Track 2 despite training only on Track 1

---

## 🏗️ Project Architecture

### High-Level Implementation Architecture

![Implementation Architecture](docs/implementation_architecture.png)
*Figure 1: High-level architecture showing the flow from simulator to Python client*

The system operates in a client-server architecture:

1. **Server (Simulator)**: Udacity Car Simulator provides three camera views (left, center, right) and telemetry data
2. **Client (Python Program)**: Deep neural network model processes images and predicts control commands
3. **Feedback Loop**: Predicted steering angles and throttle values are sent back to control the virtual car

### System Architecture

![System Architecture Diagram](docs/system_architecture_detailed.png)
*Figure 2: Detailed system architecture showing data flow and CNN processing*

**Components:**
- **Left, Center, Right Cameras**: Capture multi-perspective road images
- **Data Augmentation**: Apply random shift, rotation, and augmentation techniques
- **CNN Model**: NVIDIA architecture processes images to predict steering
- **Control Output**: Steering angle and throttle sent to simulator
- **Back-propagation**: Error feedback for weight adjustment during training

---

## ✨ Features

- **NVIDIA End-to-End CNN Architecture**: Proven architecture for autonomous driving
- **Multi-Camera Training**: Utilizes left, center, and right camera images with steering correction (±0.25)
- **Comprehensive Data Augmentation**:
  - ✅ Image cropping (remove sky and hood)
  - ✅ Horizontal flipping (eliminate directional bias)
  - ✅ Random horizontal/vertical shifts
  - ✅ Random brightness adjustment (weather generalization)
  - ✅ Random shadows (lighting conditions)
  - ✅ Random blur (camera lens simulation)
- **Real-time Autonomous Driving**: Live prediction and control via Socket.IO
- **ModelCheckpoint**: Automatic saving of best model based on validation loss
- **Track Generalization**: Model trained on Track 1 performs on Track 2

---

## 🧠 Network Architecture

### NVIDIA CNN Architecture

![NVIDIA Network Architecture](docs/nvidia_architecture.png)
*Figure 11: NVIDIA's Convolutional Neural Network architecture*

```
Input Layer: 66×200×3 RGB Image
│
├─ Normalization Layer: λx = x/127.5 - 1.0
│
├─ Convolutional Layer 1: 24 filters, 5×5 kernel, stride 2×2, ELU
│  └─ Output: Feature map 31×98
│
├─ Convolutional Layer 2: 36 filters, 5×5 kernel, stride 2×2, ELU
│  └─ Output: Feature map 14×47
│
├─ Convolutional Layer 3: 48 filters, 5×5 kernel, stride 2×2, ELU
│  └─ Output: Feature map 5×22
│
├─ Convolutional Layer 4: 64 filters, 3×3 kernel, ELU
│  └─ Output: Feature map 3×20
│
├─ Convolutional Layer 5: 64 filters, 3×3 kernel, ELU
│  └─ Output: Feature map 1×18
│
├─ Dropout Layer (0.5)
│
├─ Flatten Layer (1164 neurons)
│
├─ Fully Connected Layer 1: 100 neurons, ELU
│
├─ Fully Connected Layer 2: 50 neurons, ELU
│
├─ Fully Connected Layer 3: 10 neurons, ELU
│
└─ Output Layer: 1 neuron (Steering Angle)
```

**Total Parameters**: 348,219

### Model Implementation

```python
model = Sequential([
    Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)),
    Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
    Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
    Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
    Conv2D(64, (3, 3), activation='elu'),
    Conv2D(64, (3, 3), activation='elu'),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='elu'),
    Dense(50, activation='elu'),
    Dense(10, activation='elu'),
    Dense(1)  # Steering angle output
])
```

---

## 📦 Installation

### Requirements

```bash
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
python-socketio>=5.5.0
eventlet>=0.33.0
flask>=2.0.0
pillow>=9.0.0
```

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/AutoPilot.git
cd AutoPilot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Udacity Car Simulator**:
   - [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
   - [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
   - [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

---

## 🎮 Dataset Collection

### Simulator Interface

![Simulator First Screen](docs/first_screen.png)
*Figure 5: Simulator main menu with Training and Autonomous modes*

### Configuration

![Configuration Screen](docs/config_screen.png)
*Figure 3: Graphics and input configuration options*

![Controls Configuration](docs/controls_config.png)
*Figure 4: Keyboard/joystick control settings*

### Available Tracks

#### Track 1 (Simple)
![Track 1](docs/track1.png)
*Figure 6: Track 1 - Simple track with fewer curves, used for training*

#### Track 2 (Complex)
![Track 2](docs/track2.png)
*Figure 7: Track 2 - Complex track with high altitude, tight curves, and shadows*

### Collecting Training Data

1. **Launch the simulator** and select **TRAINING MODE**
2. **Select Track 1** for data collection
3. **Click RECORD** and choose a directory to save data
4. **Drive smoothly** for 2-3 laps:
   - Drive in center of lane
   - Include recovery maneuvers (from edges back to center)
   - Maintain consistent speed
   - Record both clockwise and counter-clockwise laps

5. **Data collected**:
   - `driving_log.csv`: Contains paths and steering angles
   - `IMG/`: Folder with three camera perspectives

### Dataset Structure

![Dataset Sample](docs/dataset_sample.png)
*Figure 8: Sample images from center, left, and right cameras*

![Driving Log CSV](docs/driving_log_csv.png)
*Figure 9: Structure of driving_log.csv file*

**CSV Columns**:
- Column 1: Center camera image path
- Column 2: Left camera image path
- Column 3: Right camera image path
- Column 4: Steering angle (0=straight, +ve=right, -ve=left)
- Column 5: Throttle (acceleration rate)
- Column 6: Brake value
- Column 7: Speed (mph)

---

## 🎨 Data Augmentation & Preprocessing

Data augmentation is crucial for generalizing the model to Track 2 (which is not used in training). The following techniques are applied:

### 1. Crop & Resize

![Crop Image](docs/crop_image.png)
*Figure 12(A): Cropping removes irrelevant sky (top 60 pixels) and hood (bottom 25 pixels)*

**Purpose**: Focus on road information by removing distractions

```python
def crop_and_resize(image):
    cropped = image[60:135, :, :]  # Remove sky and hood
    resized = cv2.resize(cropped, (200, 66))
    return resized
```

### 2. Horizontal Flip

![Flip Image](docs/flip_image.png)
*Figure 12(B): Horizontal flip eliminates left/right turn bias*

**Purpose**: Since Track 1 has more left turns, flipping creates balanced data

```python
def flip_image(image, steering_angle):
    flipped_image = cv2.flip(image, 1)
    flipped_angle = -steering_angle
    return flipped_image, flipped_angle
```

### 3. Random Shift

![Shift Vertical](docs/shift_vertical.png)
*Figure 12(C)(i): Vertical shift simulation*

![Shift Horizontal](docs/shift_horizontal.png)
*Figure 12(C)(ii): Horizontal shift with steering correction*

**Purpose**: Simulate off-center driving and recovery maneuvers

```python
def random_shift(image, steering_angle, shift_range=20):
    shift_x = np.random.randint(-shift_range, shift_range + 1)
    adjusted_angle = steering_angle + shift_x * 0.004
    # Apply transformation...
    return shifted_image, adjusted_angle
```

### 4. Brightness Adjustment

![Brightness Increased](docs/brightness.png)
*Figure 12(D): Random brightness for weather generalization*

**Purpose**: Generalize across sunny/gloomy conditions

```python
def random_brightness(image):
    brightness_factor = np.random.uniform(0.4, 1.2)
    # Adjust HSV V channel...
    return adjusted_image
```

### 5. Random Shadows

![Random Shadows](docs/random_shadows.png)
*Figure 12(E): Random polygonal shadows simulate varying lighting*

**Purpose**: Handle shadows on road and under bridges

```python
def random_shadow(image):
    # Create random shadow mask
    # Darken masked region...
    return shadowed_image
```

### 6. Random Blur

![Random Blur](docs/random_blur.png)
*Figure 12(F): Gaussian blur simulates camera lens limitations*

**Purpose**: Handle blurry camera images

```python
def random_blur(image):
    kernel_size = np.random.choice([3, 5])
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image
```

---

## 🚀 Training the Model

### Experimental Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Input Shape** | (66, 200, 3) | Height × Width × Channels |
| **Learning Rate** | 0.0001 | Adam optimizer learning rate |
| **Epochs** | 50 | Training iterations |
| **Batch Size** | 32 | Samples per batch |
| **Train/Val Split** | 80/20 | Training vs validation data |
| **Steering Correction** | ±0.25 | Left/right camera correction |
| **Dropout** | 0.5 | Regularization rate |

### Running Training

```bash
python behavioral_cloning.py
```

### Training Process

```
================================================================================
BEHAVIORAL CLONING - AUTONOMOUS DRIVING
================================================================================

[1/6] Loading driving log...
Total samples loaded: 8036

[2/6] Splitting data into training and validation sets...
Training samples: 6428
Validation samples: 1608

[3/6] Creating data generators...
Training steps per epoch: 603
Validation steps per epoch: 150

[4/6] Building NVIDIA-inspired CNN model...
Total params: 348,219

[5/6] Compiling model...
Optimizer: Adam (lr=0.0001)
Loss: MSE

[6/6] Training model...
Epoch 1/50 - loss: 0.0234 - val_loss: 0.0198
Epoch 2/50 - loss: 0.0187 - val_loss: 0.0165
...
Epoch 50/50 - loss: 0.0089 - val_loss: 0.0092

Best model saved to: model_best.h5
```

### Model Checkpoint

The `ModelCheckpoint` callback automatically saves the best model:

```python
checkpoint = ModelCheckpoint(
    'model_best.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)
```

---

## 🏁 Testing in Autonomous Mode

### Run Autonomous Driving

1. **Launch the simulator** and select **AUTONOMOUS MODE**
2. **Select a track** (Track 1 or Track 2)
3. **Run the drive script**:

```bash
python drive.py model_best.h5
```

**Optional: Set target speed**
```bash
python drive.py model_best.h5 --speed 15
```

### Console Output

```
================================================================================
AUTONOMOUS DRIVING MODE
================================================================================

Loading model: model_best.h5
Model loaded successfully!
Target speed: 9 mph
Speed limit: 15 mph

--------------------------------------------------------------------------------
Starting server...
Please start the Udacity simulator and select AUTONOMOUS MODE
--------------------------------------------------------------------------------

Connected to simulator!
Speed:   8.52 mph | Steering: -0.0234 | Throttle: 0.30
Speed:   9.15 mph | Steering:  0.0156 | Throttle: 0.10
Speed:   8.87 mph | Steering: -0.0089 | Throttle: 0.30
...
```

---

## 📊 Results

### Training Performance

![Loss Over Epochs](docs/loss_epochs.png)
*Figure 13: Training and validation loss over 50 epochs*

**Observations**:
- Initial epochs show high loss (~0.02-0.03)
- Loss rapidly decreases after epoch 10
- Validation loss stabilizes around 0.009
- No significant overfitting observed
- Best validation loss achieved: **0.0092**

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Final Training Loss** | 0.0089 |
| **Final Validation Loss** | 0.0092 |
| **Training MAE** | 0.0071 |
| **Validation MAE** | 0.0075 |
| **Track 1 Performance** | ✅ Complete lap, smooth driving |
| **Track 2 Performance** | ✅ Successful generalization |

### Generalization on Track 2

Despite training **only on Track 1**, the model successfully drives on Track 2 due to:

1. **Comprehensive augmentation** (flip, shift, brightness, shadows, blur)
2. **Multi-camera training** with steering correction
3. **Dropout regularization** prevents overfitting
4. **NVIDIA architecture** robust feature extraction

---

## 📁 Project Structure

```
AutoPilot/
│
├── behavioral_cloning.py      # Main training script
├── drive.py                    # Autonomous driving script
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── driving_log.csv            # Training data log (generated)
├── IMG/                       # Training images (generated)
│   ├── center_*.jpg
│   ├── left_*.jpg
│   └── right_*.jpg
│
├── model_best.h5              # Best trained model (generated)
├── training_history.png       # Loss plots (generated)
│
└── docs/                      # Documentation images
    ├── system_architecture.png
    ├── implementation_architecture.png
    ├── nvidia_architecture.png
    ├── first_screen.png
    ├── track1.png
    ├── track2.png
    ├── dataset_sample.png
    ├── crop_image.png
    ├── flip_image.png
    └── ...
```

---

## 🎓 Key Learnings

1. **Behavioral Cloning**: Successfully demonstrated end-to-end learning from human driving
2. **Data Augmentation**: Critical for generalization to unseen tracks
3. **Multi-Camera**: Left/right cameras with correction simulate recovery behavior
4. **NVIDIA Architecture**: Proven effective for autonomous driving tasks
5. **Overfitting Prevention**: Dropout + augmentation prevents overfitting to Track 1

---

## 🔮 Future Work

- [ ] Combine CNN (spatial features) with RNN (temporal information)
- [ ] Experiment with replacing pooling layers with recurrent layers
- [ ] Test on real-world driving datasets
- [ ] Implement transfer learning from simulator to real-world
- [ ] Add support for throttle and brake prediction
- [ ] Explore attention mechanisms for interpretability

---

## 📚 References

[1] A. Bhalla, M. S. Nikhila and P. Singh, "Simulation of Self-driving Car using Deep Learning", *3rd International Conference on Intelligent Sustainable Systems (ICISS)*, Thoothukudi, India, 2020.

[2] R. Sell, M. Leier, A. Rassõlkin and J. -P. Ernits, "Self-driving car ISEAUTO for research and education", *19th International Conference on Research and Education in Mechatronics (REM)*, Delft, Netherlands, 2018.

[3] H. Fujiyoshi, T. Hirakawa, T. Yamashita, "Deep learning-based image recognition for autonomous driving", *IATSS Research, Volume 43, Issue 4*, Kasugai, Japan, 2019.

[4] **M. Bojarski et al., "End to End Learning for Self-Driving Cars"**, NVIDIA, April 2016. [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)

[5] K. O'Shea and R. Nash, "An Introduction to Convolutional Neural Networks", 2015. [arXiv:1511.08458](https://arxiv.org/abs/1511.08458)

[6] Y. Kang, H. Yin, and C. Berger, "Test Your Self-Driving Algorithm: An Overview of Publicly Available Driving Datasets and Virtual Testing Environments", *IEEE Trans. Intell. Veh.*, vol. 4, no. 2, June 2019.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{bhujbal2023deep,
  title={Deep Learning Model for Simulating Self Driving Car},
  author={Bhujbal, Kunal and Pawar, Mahendra},
  booktitle={2023 IEEE Conference},
  year={2023},
  organization={IEEE}
}
```

---

## 📄 License

This project is for educational purposes. Udacity simulator is open-source.

---

## 🙏 Acknowledgments

- **NVIDIA** for the end-to-end learning architecture
- **Udacity** for the open-source car simulator
- **Dr. Mahendra Pawar** for guidance and supervision
- **Vasantdada Patil Pratishthan's College of Engineering & Visual Arts** for support

---

## 📞 Contact

**Kunal Bhujbal**
📧 kunalbhujbal41035@gmail.com
🏫 Vasantdada Patil Pratishthan's College of Engineering & Visual Arts, Mumbai

---

**Happy Autonomous Driving! 🚗💨**
