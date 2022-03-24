# Deep Learning Model for Simulating Self Driving Car

**End-to-end Behavioral Cloning using NVIDIA's CNN Architecture**

![System Architecture](docs/controls_config.png)

---

## Overview

This project implements an **end-to-end deep learning approach** for autonomous vehicle control using **Behavioral Cloning** in the Udacity Car Simulator. The convolutional neural network learns to predict steering angles by mimicking human driving behavior.

### Key Highlights

- Implements NVIDIA's proven end-to-end learning architecture for autonomous driving
- Multi-camera training with left, center, and right camera views
- Comprehensive data augmentation for generalization
- Successfully drives on Track 2 despite training only on Track 1

---

## Network Architecture

### NVIDIA CNN Architecture

![NVIDIA Network Architecture](docs/crop_image.png)

The model uses NVIDIA's proven architecture with 5 convolutional layers followed by 4 fully connected layers:

```
Input: 66×200×3 RGB Image
├─ Normalization Layer (x/127.5 - 1.0)
├─ Conv2D: 24 filters, 5×5, stride 2×2, ELU
├─ Conv2D: 36 filters, 5×5, stride 2×2, ELU
├─ Conv2D: 48 filters, 5×5, stride 2×2, ELU
├─ Conv2D: 64 filters, 3×3, ELU
├─ Conv2D: 64 filters, 3×3, ELU
├─ Dropout (0.5)
├─ Flatten (1164 neurons)
├─ Dense: 100 neurons, ELU
├─ Dense: 50 neurons, ELU
├─ Dense: 10 neurons, ELU
└─ Output: 1 neuron (Steering Angle)
```

**Total Parameters**: 348,219

---

## Installation

### Requirements

```bash
tensorflow>=2.8.0
keras>=2.8.0
opencv-python>=4.5.0
python-socketio>=5.5.0
eventlet>=0.33.0
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

## Dataset Collection

### Available Tracks

![Track 1](docs/track2.png)
*Track 1 - Simple track used for training*

![Track 2](docs/first_screen.png)
*Track 2 - Complex track for testing generalization*

### Collecting Data

1. Launch the simulator and select **TRAINING MODE**
2. Select **Track 1** and click **RECORD**
3. Drive smoothly for 2-3 laps (center lane + recovery maneuvers)
4. Include both clockwise and counter-clockwise laps

**Data Structure**:
- `driving_log.csv`: Contains image paths and steering angles
- `IMG/`: Three camera perspectives (left, center, right)

---

## Data Augmentation

Critical augmentation techniques for generalization:

1. **Crop & Resize**: Remove sky and hood to focus on road
2. **Horizontal Flip**: Eliminate directional bias
3. **Random Shift**: Simulate off-center driving
4. **Brightness Adjustment**: Handle different weather conditions
5. **Random Shadows**: Adapt to varying lighting
6. **Random Blur**: Simulate camera limitations

---

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Input Shape | 66×200×3 |
| Learning Rate | 0.0001 |
| Epochs | 50 |
| Batch Size | 32 |
| Train/Val Split | 80/20 |
| Steering Correction | ±0.25 |
| Dropout | 0.5 |

### Run Training

```bash
python behavioral_cloning.py
```

The best model is automatically saved as `model_best.h5` based on validation loss.

---

## Testing in Autonomous Mode

1. Launch the simulator and select **AUTONOMOUS MODE**
2. Choose a track (Track 1 or Track 2)
3. Run the drive script:

```bash
python drive.py model_best.h5
```

**Optional: Set target speed**
```bash
python drive.py model_best.h5 --speed 15
```

---

## Results

![Loss Over Epochs](docs/loss_epochs.png)

### Performance Metrics

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.0089 |
| Final Validation Loss | 0.0092 |
| Track 1 Performance | ✅ Complete lap, smooth driving |
| Track 2 Performance | ✅ Successful generalization |

The model successfully generalizes to Track 2 (unseen during training) thanks to comprehensive data augmentation and the robust NVIDIA architecture.

---

## Project Structure

```
AutoPilot/
├── behavioral_cloning.py      # Training script
├── drive.py                    # Autonomous driving script
├── requirements.txt            # Dependencies
├── driving_log.csv            # Training data log
├── IMG/                       # Training images
├── model_best.h5              # Trained model
└── docs/                      # Documentation images
```