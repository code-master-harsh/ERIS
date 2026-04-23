# ERIS — Emotion-Aware Robotic Interaction System

ERIS is a real-time multimodal system that recognises human emotion from speech and facial expressions, fuses both signals with a learned gate, generates empathetic LLM responses, and can drive an emotion-aware quadruped robot simulation.

The project exists in two tightly related layers:

1. **Multimodal emotion understanding**  
   A live Gradio application that continuously processes webcam and microphone input, predicts emotion, tracks emotional context, and generates short supportive responses.

2. **Embodied robotic expression**  
   A ROS 2 / Gazebo / CHAMP-based quadruped control stack that converts the predicted emotion into physically distinct gait, posture, and motion behaviour.

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Emotion Classes](#emotion-classes)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Robot Integration](#robot-integration)
- [Model Notes](#model-notes)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

ERIS started as a multimodal emotion recognition system and was extended into a complete interaction pipeline. The final system can:

- capture **live audio and video**
- predict one of four emotions: **angry, happy, neutral, sad**
- combine audio and visual evidence using a **learned gated fusion module**
- keep short-term emotional context across successive predictions
- generate empathetic responses with an LLM
- export the predicted emotional state to a robot simulation
- modulate a quadruped’s gait and posture to match the detected emotion

The goal is not only to classify emotion, but to make the response feel more natural, contextual, and embodied.

---

## Core Features

### Live multimodal emotion analysis
- Continuous webcam and microphone capture in background threads
- Rolling multimodal windows for repeated inference
- Real-time prediction display
- Emotion timeline / history across the session
- Live gate readout showing the balance between audio and visual modalities

### Empathetic conversational response
- Context-aware short replies generated through Groq-hosted Llama 3.1 8B
- Response behaviour that can adapt to:
  - the current emotion
  - the detected intensity
  - recent emotion shifts
- Optional response suppression when the emotional state is stable or not strong enough to require a reply

### Offline / upload mode
- Analyse a pre-recorded video file
- Run the same multimodal inference pipeline
- Display class probabilities and gate behaviour
- Skip the live conversational layer when not needed

### Robot expression
- Emotion state written to a shared file for the robot process
- Quadruped gait changes based on the predicted emotion
- Smooth transitions between states using cosine interpolation
- Emotion-specific posture, velocity, and motion profiles

---

## System Architecture

```text
Audio stream (microphone)
        │
        ▼
   WavLM-Base+
        │
   attention pooling
        │
        ├───────────────┐
        │               │
        ▼               ▼
   Learned gate     ViT-Face-Expression
        │               │
        └───────┬───────┘
                ▼
        Multimodal classifier
                ▼
     angry / happy / neutral / sad
                ▼
   ┌────────────┴────────────┐
   ▼                         ▼
Empathetic LLM reply     Robot emotion state
   ▼                         ▼
User-facing response    Quadruped gait control
```

---

## Emotion Classes

ERIS currently predicts four emotions:

- **angry**
- **happy**
- **neutral**
- **sad**

These classes are used consistently across the multimodal model, the live response layer, and the robot behaviour mapping.

---

## Project Structure

A typical project layout is:

```text
ERIS/
├── app.py                         # Gradio UI, live inference, LLM integration
├── model.py                       # Multimodal model architecture
├── multimodal_final_v4.pt         # Trained model checkpoint
├── emotion_state.json             # Shared emotion state for the robot
├── blaze_face_short_range.tflite   # MediaPipe face detector model
├── .env                           # API keys and local configuration
├── .gitignore
├── README.md
├── Stride_bot-main/               # Robot simulation / gait control
│   ├── bot_walk.py                # Robot runtime loop
│   ├── stridebot.urdf             # Robot description
│   ├── src/
│   │   ├── emotion_gait.py        # Emotion-specific gait parameters
│   │   ├── gaitPlanner.py         # Base trot gait planner
│   │   ├── kinematic_model.py     # IK model
│   │   └── ...
│   └── LICENSE
└── ros2_quad_ws/ or similar       # ROS 2 integration workspace, if used
    ├── src/
    ├── launch/
    └── ...
```

---

## Requirements

### For the multimodal app
- Python 3.10 or newer
- `ffmpeg`
- Webcam
- Microphone

### For the robot integration stack
- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Fortress
- CHAMP-based quadruped control stack
- PyTorch, OpenCV, Gradio, MediaPipe, SoundDevice

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ERIS
```

### 2. Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install torch torchvision transformers mediapipe opencv-python \
            sounddevice soundfile gradio groq python-dotenv \
            matplotlib pybullet numpy
```

### 4. Download the MediaPipe face detector model

```bash
curl -L -o blaze_face_short_range.tflite \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
```

### 5. Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

---

## Running the Project

### Multimodal emotion recognition app

```bash
python app.py
```

The Gradio app runs locally and provides a live interface for webcam and microphone input.

### Robot companion

In a separate terminal, run the robot simulation:

```bash
cd Stride_bot-main
python bot_walk.py
```

The robot reads the current emotion from `emotion_state.json` and changes gait accordingly.

### ROS 2 / Gazebo behaviour stack

If you are using the ROS 2 integration variant, start the simulator and behaviour nodes in separate terminals:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch go2_config gazebo.launch.py
ros2 launch go2_behaviour emotion_motion.launch.py
```

You can also dispatch manual commands from the CLI:

```bash
ros2 run go2_behaviour behaviour_commander --action move --value 3.0 --emotion happy
ros2 run go2_behaviour behaviour_commander --action turn --value 90.0 --emotion angry
ros2 run go2_behaviour behaviour_commander --action move --value 1.5 --emotion sad
```

---

## Robot Integration

The robot side of ERIS is designed to turn emotion into visible motion rather than a simple preset animation.

### Behaviour mapping
- **Happy**: energetic, bouncy, prancing gait
- **Sad**: slower movement with a hunched posture
- **Angry**: faster, more forceful motion with sharper turns
- **Neutral**: stable, calm baseline locomotion

### Motion design
- Smooth interpolation between emotion states
- Safe posture updates to avoid inverse kinematics instability
- Dynamic adjustment of gait parameters such as stance duration and swing height
- Heading correction using ground-truth odometry in the ROS 2 variant

### Bridge between app and robot
The multimodal app can publish the current emotion state directly to the robot control layer, allowing the perception pipeline and the robot behaviour pipeline to operate as one connected system.

---

## Model Notes

The multimodal network uses:

- **WavLM-Base+** for speech emotion features
- **ViT-Face-Expression** for facial emotion features
- **Temporal pooling / transformer-based aggregation** for sequence-level video understanding
- **Learned gated fusion** to balance audio and video on a per-sample basis
- **Final classifier** to predict the emotion label

This design makes the model more robust than using either modality alone.

---

## Acknowledgements

- **WavLM-Base+** for speech feature extraction
- **ViT-Face-Expression** for facial emotion classification
- **Groq / Llama 3.1 8B** for low-latency empathetic response generation
- **MediaPipe** for face detection
- **Stride Bot** for the quadruped simulation and gait framework
- **ROS 2**, **Gazebo Fortress**, and **CHAMP** for the embodied robotics pipeline

---

## License

This project contains components derived from the Stride Bot quadruped simulation, which retains its original GPL-3.0 licence in `Stride_bot-main/LICENSE`. Any additional project code should be licensed according to your repository policy.

---

## Summary

ERIS is a real-time emotion-aware interaction system that combines speech and facial emotion recognition, learned multimodal fusion, contextual LLM responses, and emotion-driven quadruped motion into a single integrated pipeline.
