# ERIS — Emotion-Aware Robotic Interaction System

Real-time multimodal emotion recognition that combines **speech** and **facial expression** analysis with an empathetic LLM companion and an emotion-driven quadruped robot.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multimodal Emotion Model                     │
│                                                                 │
│  Audio (wav) ──► WavLM-Base+ ──► Attention Pooling ──► 768d     │
│                                                         │       │
│                                                    GatedFusion  │
│                                                         │       │
│  Video (T,C,H,W) ──► ViT-Face-Expression ──►           │        │
│       + Confidence-Weighted Temporal Pooling ──► 768d   │       │
│                                                         ▼       │
│                                              Classifier ──► 4   │
│                                         (angry, happy,          │
│                                          neutral, sad)          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Model / Method | Purpose |
|-----------|---------------|---------|
| **SER** (Speech) | [WavLM-Base+](https://huggingface.co/microsoft/wavlm-base-plus) | Audio feature extraction with weighted layer fusion + attention pooling |
| **FER** (Face) | [ViT-Face-Expression](https://huggingface.co/trpakov/vit-face-expression) | Per-frame face emotion classification with confidence-weighted vote aggregation |
| **Fusion** | Deep MLP Gated Fusion | Learned gate balances audio vs. visual modalities per-sample |
| **LLM** | [Llama 3.1 8B](https://groq.com/) via Groq | Empathetic real-time responses based on emotional state tracking |
| **Robot** | [Stride Bot](Stride_bot-main/) (PyBullet) | Quadruped robot that walks with emotion-specific gaits |

## Features

### 🎙️ Live Analysis
- Continuous webcam + microphone capture in background threads
- Rolling 4-second window with predictions every 5 seconds
- Real-time emotion timeline visualization
- Empathetic LLM responses via Groq (Llama 3.1 8B) that:
  - Detect emotional shifts and ask what happened
  - Adapt response tone to emotion intensity
  - Maintain conversation context across the session

### 📁 Video Upload
- One-shot multimodal analysis of pre-recorded videos
- Same model pipeline, no LLM responses

### 🤖 Robot Companion
- PyBullet quadruped robot driven by predicted emotions
- **Happy**: Bouncy, energetic trot with head wagging
- **Sad**: Slow, hunched walk with drooped posture
- **Angry**: Fast, aggressive stride with rotational movement
- **Neutral**: Calm standing with gentle sway
- Smooth 3-second cosine-interpolated transitions between emotion states

## Setup

### Prerequisites

- Python 3.11+
- ffmpeg (for audio/video muxing)
- Webcam + microphone

### Installation

```bash
# Install dependencies
pip install torch torchvision transformers mediapipe opencv-python \
            sounddevice soundfile gradio groq python-dotenv \
            matplotlib pybullet numpy

# Download MediaPipe face detector model
curl -L -o blaze_face_short_range.tflite \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
```

### Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Running

```bash
# Start the Gradio web interface
python app.py

# (Optional) Start the robot companion in a separate terminal
cd Stride_bot-main
python bot_walk.py
```

The app runs at `http://localhost:7860`. The robot reads emotions from `emotion_state.json` written by the app.

## Project Structure

```
emotion_v4-copy/
├── app.py                      # Gradio UI + live inference + LLM integration
├── model.py                    # Model architecture (SER, FER, GatedFusion)
├── multimodal_final_v4.pt      # Trained model checkpoint
├── blaze_face_short_range.tflite  # MediaPipe face detector
├── emotion_state.json          # Shared state file (app → robot)
├── .env                        # API keys (not committed)
├── .gitignore
└── Stride_bot-main/            # Robot simulation (GPL-3.0)
    ├── bot_walk.py             # Main robot loop (modified)
    ├── stridebot.urdf          # Robot model
    ├── LICENSE                 # GPL-3.0 (original)
    ├── src/
    │   ├── emotion_gait.py     # Emotion-specific gait parameters (new)
    │   ├── gaitPlanner.py      # Trot gait planner (original)
    │   ├── kinematic_model.py  # Inverse kinematics (original)
    │   ├── pybullet_debugger.py
    │   ├── IK_solver.py
    │   └── geometrics.py
    └── STL_simulation/         # Robot 3D models
```

## Acknowledgements

- **Stride Bot** — Quadruped simulation by [Team STRIDE](https://github.com/nicrusso7/rex-gym), licensed under GPL-3.0. The original project provides the robot URDF, kinematics, and gait planner. We modified `bot_walk.py` to read emotions from the recognition system and added `src/emotion_gait.py` for emotion-specific gait parameters.
- **WavLM** — [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) for speech feature extraction.
- **ViT-Face-Expression** — [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) for facial emotion classification.
- **Groq** — Ultra-low-latency LLM inference for real-time empathetic responses.
- **MediaPipe** — Google's face detection pipeline.

## Changes to Stride Bot

The following modifications were made to the original Stride Bot codebase:

| File | Change |
|------|--------|
| `bot_walk.py` | Replaced hardcoded emotion with live JSON file reader; added smooth cosine-interpolated transitions between emotion states; added velocity safety clamp |
| `src/emotion_gait.py` | **New file** — defines gait parameters (stride length, step period, body pitch, etc.) for each emotion |

The original LICENSE (GPL-3.0) is preserved in `Stride_bot-main/LICENSE`.
