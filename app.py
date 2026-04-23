"""
Multimodal Emotion Recognition v4 — Gradio LIVE UI (always-on capture)
Continuously streams webcam + mic in background threads, shows a live mirror
of the user's own video, and runs inference on a rolling 4-second clip every
few seconds — predict() is unchanged from the one-shot version.
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from collections import deque

# Load .env BEFORE any HuggingFace imports (model.py needs HF_TOKEN)
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

import cv2
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import gradio as gr
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

from model import SERModel, FERModel, MultimodalEmotionModel
from groq import Groq
import collections
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import concurrent.futures
import re

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.1-8b-instant"   

# =========================================================
# EMOTIONAL STATE TRACKER + CONTEXT ENGINE + LLM RESPONSE
# =========================================================

@dataclass
class EmotionSnapshot:
    label      : str
    confidence : float
    gate       : float
    timestamp  : float = field(default_factory=time.time)


class EmotionalStateTracker:
    def __init__(self, window: int = 3):
        self.history               : collections.deque = collections.deque(maxlen=window)
        self.last_response_emotion : Optional[str]     = None
        self.last_response_time    : float             = 0.0
        self.previous_emotion      : Optional[str]     = None

    def update(self, label: str, confidence: float, gate: float):
        if self.history and self.history[-1].label != label:
            self.previous_emotion = self.history[-1].label
        self.history.append(EmotionSnapshot(label, confidence, gate))

    @property
    def dominant_emotion(self) -> Optional[str]:
        if not self.history:
            return None
        return collections.Counter(s.label for s in self.history).most_common(1)[0][0]

    @property
    def is_stable(self) -> bool:
        if len(self.history) < 2:
            return False
        return len(set(s.label for s in list(self.history)[-2:])) == 1

    @property
    def just_shifted(self) -> bool:
        if len(self.history) < 2:
            return False
        h = list(self.history)
        return h[-2].label != h[-1].label

    @property
    def mean_confidence(self) -> float:
        if not self.history:
            return 0.0
        return sum(s.confidence for s in self.history) / len(self.history)

    @property
    def intensity(self) -> str:
        c = self.mean_confidence
        if c > 0.75: return "high"
        if c > 0.50: return "medium"
        return "low"

    @property
    def dominant_modality(self) -> str:
        if not self.history:
            return "balanced"
        avg = sum(s.gate for s in self.history) / len(self.history)
        if avg > 0.55: return "voice"
        if avg < 0.45: return "face"
        return "both face and voice"

    def should_respond(self, cooldown_secs: float = 8.0) -> bool:
        now      = time.time()
        cooldown = (now - self.last_response_time) > cooldown_secs
        shifted  = self.dominant_emotion != self.last_response_emotion
        if self.just_shifted and (now - self.last_response_time) > 3.0:
            return True
        return self.is_stable and (cooldown or shifted)

    def mark_responded(self):
        self.last_response_time    = time.time()
        self.last_response_emotion = self.dominant_emotion


class ResponseType(Enum):
    NONE        = "none"
    ACKNOWLEDGE = "acknowledge"
    EMPATHISE   = "empathise"
    ENCOURAGE   = "encourage"
    CHECK_IN    = "check_in"
    SHIFT_NOTED = "shift_noted"


def decide_response_type(tracker: EmotionalStateTracker) -> ResponseType:
    if tracker.just_shifted and len(tracker.history) >= 2:
        if tracker.should_respond():
            return ResponseType.SHIFT_NOTED

    if not tracker.should_respond():
        return ResponseType.NONE

    emotion = tracker.dominant_emotion

    if emotion == "neutral":
        if len(tracker.history) == tracker.history.maxlen:
            if all(s.label == "neutral" for s in tracker.history):
                return ResponseType.CHECK_IN

    if emotion in ("angry", "sad"):
        return ResponseType.EMPATHISE if tracker.intensity == "high" \
               else ResponseType.ACKNOWLEDGE

    if emotion == "happy":
        return ResponseType.ENCOURAGE

    return ResponseType.ACKNOWLEDGE

_conversation_history : list[dict]                      = []
_pending_response     : Optional[concurrent.futures.Future] = None
_groq_executor        = concurrent.futures.ThreadPoolExecutor(max_workers=1)
MAX_HISTORY           = 6

def generate_response(tracker: EmotionalStateTracker,
                      response_type: ResponseType) -> Optional[str]:
    if response_type == ResponseType.NONE:
        return None

    emotion  = tracker.dominant_emotion
    intensity = tracker.intensity
    modality  = tracker.dominant_modality
    shifted   = tracker.just_shifted

    system_prompt = """\
You are a calm, empathetic emotion-aware assistant embedded in a real-time \
emotion recognition system.

Your role is to acknowledge the user's detected emotional state in a way \
that feels natural and supportive — like a thoughtful colleague, not a therapist.

Rules you must always follow:
- Keep responses to 1-2 sentences maximum. Never longer.
- Never say "I detected" or "the system shows" — speak directly to the user.
- Never repeat a response you have already given in this conversation.
- Do not ask multiple questions in one response. One question maximum.
- Match your tone to the intensity: low intensity = gentle observation, \
high intensity = more direct empathy.
- If the emotion is neutral, keep it light — do not over-interpret calmness.
- Never use bullet points, lists, or markdown formatting.
- Sound like a human, not an AI assistant.
- When the user's emotion has just shifted drastically (e.g., happy to sad), \
acknowledge the change itself — ask gently what happened rather than just \
responding to the new emotion in isolation."""

    shift_context = ""
    if response_type == ResponseType.SHIFT_NOTED and tracker.previous_emotion:
        shift_context = f"\n- Shifted FROM: {tracker.previous_emotion} \u2192 TO: {emotion}"
        shift_context += "\n- IMPORTANT: Acknowledge this sudden change. Ask what happened."

    context_message = f"""\
Current emotional context:
- Detected emotion: {emotion}
- Intensity: {intensity}
- Detected primarily through: {modality}
- Emotion just shifted from previous state: {shifted}{shift_context}
- Response type needed: {response_type.value}
- Recent emotion history: {[s.label for s in tracker.history]}

Generate a single natural response appropriate for this context."""

    messages = (
        [{"role": "system", "content": system_prompt}]
        + _conversation_history[-MAX_HISTORY:]
        + [{"role": "user", "content": context_message}]
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=80,
            top_p=0.9,
        )
        response = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq] API call failed: {e}")
        return None

    _conversation_history.append({"role": "user",      "content": context_message})
    _conversation_history.append({"role": "assistant",  "content": response})
    tracker.mark_responded()
    return response

tracker = EmotionalStateTracker(window=3)

def _parse_gate_value(gate_desc: str) -> float:
    match = re.search(r"([0-9]+\.[0-9]+)", gate_desc)
    return float(match.group(1)) if match else 0.5

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "multimodal_final_v4.pt"
SAMPLE_RATE = 16_000
MAX_DURATION = 3.5
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

NUM_FRAMES = 16
IMG_SIZE = 224
VISUAL_DIM = 768

EMOTIONS = ["angry", "happy", "neutral", "sad"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEW: vit-face-expression requires 0.5 normalization ---
_IMAGENET_MEAN = [0.5, 0.5, 0.5]
_IMAGENET_STD = [0.5, 0.5, 0.5]

# --- LIVE CAPTURE ---
CLIP_SECONDS       = 4.0   
CAM_FPS            = 20     
MIRROR_FPS         = 10     
TICK_INTERVAL      = 5.0    
HISTORY_SECONDS    = 60     
WEBCAM_INDEX       = 0

# ── UI: pastel green accent ──────────────────────────────
EMOTION_COLORS = {
    "angry":   "#e07070",
    "happy":   "#a8c5a0",
    "neutral": "#8a857a",
    "sad":     "#6b9bd1",
}


# =========================================================
# AUDIO PIPELINE
# =========================================================
def _resample_linear(wav, orig_sr, target_sr):
    if orig_sr == target_sr or len(wav) < 2:
        return wav.astype(np.float32)
    target_len = int(round(len(wav) * target_sr / orig_sr))
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)

def load_wav_from_file(path, sr=SAMPLE_RATE):
    try:
        wav, orig_sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return _resample_linear(wav, orig_sr, sr)
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
             "-f", "wav", "-acodec", "pcm_s16le", tmp_path],
            check=True, capture_output=True,
        )
        wav, _ = sf.read(tmp_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def vad_trim(wav, thr=0.01, frame_ms=20):
    hop = int(SAMPLE_RATE * frame_ms / 1000)
    if len(wav) < hop:
        return wav
    rms = np.array([
        np.sqrt(np.mean(wav[i:i + hop] ** 2))
        for i in range(0, len(wav), hop)
    ])
    if len(rms) == 0 or rms.max() <= 1e-8:
        return wav
    mask = rms > thr * rms.max()
    if mask.sum() == 0:
        return wav
    idx = np.where(mask)[0]
    return wav[idx[0] * hop : min(len(wav), idx[-1] * hop + hop)]

def normalize_amplitude(wav, peak=0.95):
    mx = np.abs(wav).max()
    return wav * (peak / mx) if mx > 1e-6 else wav

def segment_and_pad(wav, length=MAX_SAMPLES):
    if len(wav) >= length:
        start = (len(wav) - length) // 2
        return wav[start:start + length]
    return np.pad(wav, (0, length - len(wav)))

def preprocess_audio(path):
    try:
        wav = load_wav_from_file(path)
        wav = vad_trim(wav)
        wav = normalize_amplitude(wav)
        return segment_and_pad(wav)
    except Exception as e:
        print(f"[audio] failed: {e}")
        return None


# =========================================================
# FACE / CLIP PIPELINE
# =========================================================
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_tflite_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
_base_options = mp_python.BaseOptions(model_asset_path=_tflite_path)
_options = mp_vision.FaceDetectorOptions(base_options=_base_options, min_detection_confidence=0.3)
_face_detector = mp_vision.FaceDetector.create_from_options(_options)

def _clip_eval_transform(frames):
    processed = []
    for f in frames:
        f = TF.resize(f, (IMG_SIZE, IMG_SIZE),
                      interpolation=TF.InterpolationMode.BICUBIC)
        t = TF.to_tensor(f)
        t = TF.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
        processed.append(t)
    return torch.stack(processed)

def extract_face_crop(frame_bgr, margin_frac=0.25, min_confidence=0.3):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _face_detector.detect(mp_image)
    if not res.detections:
        return None, 0.0

    det = res.detections[0]
    score = det.categories[0].score if det.categories else 0.0
    bbox = det.bounding_box
    
    ih, iw = frame_bgr.shape[:2]
    
    x = bbox.origin_x
    y = bbox.origin_y
    w = bbox.width
    h = bbox.height
    
    mx = int(w * margin_frac)
    my = int(h * margin_frac)
    
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(iw, x + w + mx), min(ih, y + h + my)
    if (x2 - x1) < 40 or (y2 - y1) < 40:
        return None, 0.0

    return Image.fromarray(rgb[y1:y2, x1:x2]), score

def read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames

def build_clip_from_video(video_path, num_frames=NUM_FRAMES):
    frames_bgr = read_all_frames(video_path)
    if len(frames_bgr) == 0:
        return [], [], 0    # NEW: returning empty scores list
        
    total = len(frames_bgr)
    positions = np.linspace(0.1, 0.9, num_frames)
    sampled_idx = [min(int(total * p), total - 1) for p in positions]

    faces = []
    scores = []             # NEW: Track scores
    for idx in sampled_idx:
        face, score = extract_face_crop(frames_bgr[idx])
        if face is not None:
            faces.append(face)
            scores.append(score)

    n_detected = len(faces)
    if n_detected < num_frames // 2:
        return [], [], n_detected
        
    # Pad if we have fewer than num_frames
    while len(faces) < num_frames:
        faces.append(faces[-1])
        scores.append(scores[-1]) 
        
    return faces[:num_frames], scores[:num_frames], n_detected


# =========================================================
# MODEL LOAD
# =========================================================
def load_model():
    ser = SERModel(n_class=4, load_pretrained=False)
    fer = FERModel(
        n_class=4, n_frames=NUM_FRAMES,
        embed_dim=VISUAL_DIM, img_size=IMG_SIZE,
        load_pretrained=False,
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    net = MultimodalEmotionModel(
        ser=ser, fer=fer,
        fusion_dim=cfg.get("fusion_dim", 512),
        n_classes=cfg.get("n_classes", 4),
        visual_dim=cfg.get("visual_dim", VISUAL_DIM),
        dropout=cfg.get("dropout", 0.3),
    )
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.to(DEVICE).eval()
    labels = cfg.get("emotion_labels", EMOTIONS)
    n_frames = cfg.get("num_frames", NUM_FRAMES)
    return net, labels, ckpt.get("results", {}), n_frames


print(f"[model] loading on {DEVICE}...")
MODEL, EMOTION_LABELS, RESULTS, CKPT_NUM_FRAMES = load_model()
print(f"[model] ready. labels = {EMOTION_LABELS}, n_frames = {CKPT_NUM_FRAMES}")


# =========================================================
# predict()
# =========================================================
def predict(video_path):
    empty = {label: 0.0 for label in EMOTION_LABELS}
    if video_path is None or not os.path.exists(video_path):
        return empty, "—", "awaiting input", ""

    # NEW: Unpack scores
    faces, scores, n_detected = build_clip_from_video(video_path, num_frames=CKPT_NUM_FRAMES)
    audio = preprocess_audio(video_path)

    if not faces:
        return empty, "—", "no face detected", \
            f"only {n_detected}/{CKPT_NUM_FRAMES // 2} frames had a face"
    if audio is None:
        return empty, "—", "audio extraction failed", ""

    wav_tensor = torch.from_numpy(audio).unsqueeze(0).to(DEVICE)
    clip_tensor = _clip_eval_transform(faces).unsqueeze(0).to(DEVICE)
    
    # NEW: Convert scores to a tensor
    scores_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # NEW: Pass img_scores to the model
        logits, gate = MODEL(wav_tensor, clip_tensor, img_scores=scores_tensor, return_gate=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        gate_mean = float(gate.mean().item())

    probs_dict = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
    pred_label = EMOTION_LABELS[int(np.argmax(probs))]

    if gate_mean > 0.55:
        gate_desc = f"{gate_mean:.3f} · leans audio"
    elif gate_mean < 0.45:
        gate_desc = f"{gate_mean:.3f} · leans visual"
    else:
        gate_desc = f"{gate_mean:.3f} · balanced"

    # Status updated to reflect the new architecture
    status = f"{n_detected}/{CKPT_NUM_FRAMES} frames detected · ViT-Face-Expression + Weighted Pooling"
    return probs_dict, pred_label, gate_desc, status


# =========================================================
# CONTINUOUS BACKGROUND CAPTURE
# =========================================================
_cam_lock = threading.Lock()
_aud_lock = threading.Lock()

FRAME_BUFFER = deque(maxlen=int((CLIP_SECONDS + 1.0) * CAM_FPS))
AUDIO_BUFFER_MAXLEN = int((CLIP_SECONDS + 1.0) * SAMPLE_RATE)
AUDIO_BUFFER = deque(maxlen=AUDIO_BUFFER_MAXLEN)

_capture_running = threading.Event()
_cam_thread = {"t": None}
_aud_stream = {"s": None}
_cam_frame_shape = {"wh": (640, 480)}

def _cam_loop():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[cam] could not open webcam")
        _capture_running.clear()
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    _cam_frame_shape["wh"] = (w, h)

    interval = 1.0 / CAM_FPS
    next_t = time.time()
    while _capture_running.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        ts = time.time()
        with _cam_lock:
            FRAME_BUFFER.append((ts, frame))
        next_t += interval
        sleep_for = next_t - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_t = time.time()

    cap.release()

def _audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0] if indata.ndim > 1 else indata
    with _aud_lock:
        AUDIO_BUFFER.extend(mono.tolist())

def start_capture_threads():
    if _capture_running.is_set():
        return True
    _capture_running.set()

    t = threading.Thread(target=_cam_loop, daemon=True)
    _cam_thread["t"] = t
    t.start()

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_audio_callback,
            blocksize=1024,
        )
        stream.start()
        _aud_stream["s"] = stream
    except Exception as e:
        print(f"[audio] stream failed: {e}")
        _aud_stream["s"] = None

    time.sleep(0.5)
    return True

def stop_capture_threads():
    _capture_running.clear()
    if _aud_stream["s"] is not None:
        try:
            _aud_stream["s"].stop()
            _aud_stream["s"].close()
        except Exception:
            pass
        _aud_stream["s"] = None
    with _cam_lock:
        FRAME_BUFFER.clear()
    with _aud_lock:
        AUDIO_BUFFER.clear()

def snapshot_clip_to_mp4(seconds=CLIP_SECONDS, fps=CAM_FPS, sr=SAMPLE_RATE):
    with _cam_lock:
        if len(FRAME_BUFFER) == 0:
            return None
        newest_t = FRAME_BUFFER[-1][0]
        cutoff = newest_t - seconds
        clip_frames = [(ts, f.copy()) for (ts, f) in FRAME_BUFFER if ts >= cutoff]

    if len(clip_frames) < max(4, int(seconds * fps) // 4):
        return None

    with _aud_lock:
        audio_samples = np.array(AUDIO_BUFFER, dtype=np.float32)
    target_len = int(seconds * sr)
    if len(audio_samples) >= target_len:
        audio_samples = audio_samples[-target_len:]

    w, h = _cam_frame_shape["wh"]
    tmp_video = tempfile.NamedTemporaryFile(suffix="_v.mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    for _, frame in clip_frames:
        writer.write(frame)
    writer.release()

    tmp_audio = None
    if audio_samples.size > 0:
        tmp_audio = tempfile.NamedTemporaryFile(suffix="_a.wav", delete=False).name
        sf.write(tmp_audio, audio_samples, sr, subtype="PCM_16")

    tmp_final = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        if tmp_audio is not None:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", tmp_video, "-i", tmp_audio,
                 "-c:v", "copy", "-c:a", "aac", "-shortest",
                 tmp_final],
                check=True, capture_output=True,
            )
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", tmp_video, "-c:v", "copy", tmp_final],
                check=True, capture_output=True,
            )
    except Exception as e:
        print(f"[mux] ffmpeg failed: {e}")
        if os.path.exists(tmp_final):
            try: os.remove(tmp_final)
            except Exception: pass
        tmp_final = tmp_video

    for p in (tmp_video, tmp_audio):
        if p and p != tmp_final and os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    return tmp_final


# =========================================================
# HISTORY & TIMELINE PLOT
# =========================================================
HISTORY = deque(maxlen=500)
SESSION_START = {"t": None}

def _make_timeline_plot():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)

    bg   = "#1a1a18"
    grid = "#2d2b27"
    ink  = "#f5f2ea"
    ink_dim = "#8a857a"
    accent  = "#a8c5a0"   

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if len(HISTORY) == 0:
        ax.text(0.5, 0.5, "awaiting data",
                ha="center", va="center",
                color=ink_dim, fontsize=14,
                fontstyle="italic", family="serif",
                transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        return fig

    times = np.array([h[0] for h in HISTORY])
    for emo in EMOTION_LABELS:
        ys = np.array([h[1].get(emo, 0.0) for h in HISTORY])
        ax.plot(times, ys,
                color=EMOTION_COLORS.get(emo, accent),
                linewidth=2.0, marker="o", markersize=4,
                label=emo)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("seconds", color=ink_dim, fontsize=10, family="monospace")
    ax.set_ylabel("confidence", color=ink_dim, fontsize=10, family="monospace")
    ax.tick_params(colors=ink_dim, labelsize=9)
    ax.grid(True, color=grid, linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(grid); spine.set_linewidth(0.8)

    leg = ax.legend(
        loc="upper left", ncol=4, frameon=False, fontsize=10,
        labelcolor=ink, prop={"family": "monospace"},
        bbox_to_anchor=(0.0, 1.18),
    )
    for t in leg.get_texts():
        t.set_color(ink)

    xmax = float(times[-1])
    xmin = max(0.0, xmax - HISTORY_SECONDS)
    ax.set_xlim(xmin, max(xmax, xmin + 1))

    plt.tight_layout()
    return fig


# =========================================================
# UI helpers
# =========================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,400;1,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0f0f0e;
  --bg-card: #1a1a18;
  --bg-elev: #242421;
  --ink: #f5f2ea;
  --ink-dim: #8a857a;
  --ink-faint: #4a4740;
  --accent: #a8c5a0;
  --accent-hover: #bdd4b8;
  --rule: #2d2b27;
}

* { box-sizing: border-box; }
html, body, .gradio-container { background: var(--bg) !important; }

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  font-family: 'Fraunces', serif !important;
  color: var(--ink) !important;
  padding: 48px 32px !important;
}

.hero {
  padding: 0 0 40px 0;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 40px;
  text-align: center;
  width: 100%;
}
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--accent);
  margin-bottom: 20px;
}
.hero-title {
  font-family: 'Fraunces', serif; font-weight: 300;
  font-size: clamp(36px, 5vw, 64px); line-height: 1.05;
  letter-spacing: -0.025em; color: var(--ink);
  margin: 0 0 16px 0;
}
.hero-title em { font-style: italic; font-weight: 400; color: var(--accent); }
.hero-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; color: var(--ink-dim);
  line-height: 1.7;
  max-width: 100%;
  margin: 0 auto;
}

.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 0.24em;
  text-transform: uppercase; color: var(--ink-dim);
  margin: 0 0 14px 0;
  display: flex; align-items: center; gap: 12px;
}
.section-label::before {
  content: ''; width: 28px; height: 1px;
  background: var(--accent); display: inline-block;
}

.primary button, button.primary, .gr-button-primary {
  background: var(--accent) !important; color: var(--bg) !important;
  border: none !important; border-radius: 2px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important; font-weight: 500 !important;
  letter-spacing: 0.14em !important; text-transform: uppercase !important;
  padding: 14px 28px !important; transition: all 0.2s ease !important;
  box-shadow: none !important;
}
.primary button:hover, button.primary:hover {
  background: var(--accent-hover) !important; transform: translateY(-1px);
}
.secondary button, button.secondary {
  background: transparent !important; color: var(--ink-dim) !important;
  border: 1px solid var(--rule) !important; border-radius: 2px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important; letter-spacing: 0.14em !important;
  text-transform: uppercase !important; padding: 14px 24px !important;
}
.secondary button:hover, button.secondary:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}

.prediction-card {
  background: var(--bg-card); border: 1px solid var(--rule);
  border-left: 2px solid var(--accent);
  padding: 36px 30px; border-radius: 2px;
  min-height: 220px;
  display: flex; flex-direction: column; justify-content: center;
}
.prediction-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 0.24em;
  text-transform: uppercase; color: var(--ink-dim);
  margin-bottom: 14px;
}
.prediction-value {
  font-family: 'Fraunces', serif; font-style: italic;
  font-weight: 400; font-size: 72px; line-height: 1;
  letter-spacing: -0.025em; color: var(--ink); margin: 0;
}
.prediction-placeholder {
  font-family: 'Fraunces', serif; font-style: italic;
  font-weight: 300; font-size: 42px;
  color: var(--ink-faint); letter-spacing: -0.02em;
}

.gate-row {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 16px 22px; margin-top: 12px;
  background: var(--bg-card); border: 1px solid var(--rule); border-radius: 2px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
}
.gate-label { color: var(--ink-dim); letter-spacing: 0.18em; text-transform: uppercase; font-size: 10px; }
.gate-value { color: var(--accent); font-weight: 500; letter-spacing: 0.04em; }

.metrics-strip {
  display: flex; gap: 0;
  border: 1px solid var(--rule); border-radius: 2px;
  overflow: hidden; background: var(--bg-card);
  margin-top: 16px;
}
.metric { flex: 1; padding: 22px 24px; border-right: 1px solid var(--rule); }
.metric:last-child { border-right: none; }
.metric-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--ink-dim); margin-bottom: 8px;
}
.metric-value {
  font-family: 'Fraunces', serif; font-style: italic;
  font-weight: 400; font-size: 32px; color: var(--ink);
  line-height: 1; letter-spacing: -0.02em;
}
.metric-value .unit { font-size: 14px; color: var(--ink-dim); font-style: normal; margin-left: 2px; }
.metric--accent .metric-value { color: var(--accent); }

.mirror, .mirror > div {
  background: var(--bg-card) !important; border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
}
.mirror img { border-radius: 2px !important; background: var(--bg-elev) !important; }

.output-class, .gr-label, .label-container {
  background: var(--bg-card) !important; border: 1px solid var(--rule) !important;
  border-radius: 2px !important; padding: 20px !important;
}
.label-wrap, .confidence-set { font-family: 'JetBrains Mono', monospace !important; }
.label-wrap .confidence-set .bar, .gr-label .bar {
  background: var(--accent) !important; border-radius: 0 !important;
}

.status-line {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--ink-dim); letter-spacing: 0.06em;
  padding: 14px 0 0 0; border-top: 1px dashed var(--rule); margin-top: 18px;
}
.status-line::before { content: '› '; color: var(--accent); }

.live-dot {
  display: inline-block; width: 8px; height: 8px;
  background: var(--accent); border-radius: 50%;
  margin-right: 8px; box-shadow: 0 0 10px var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
  vertical-align: middle;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.85); }
}

.button-row { margin-top: 16px !important; gap: 12px !important; }

footer, .show-api, .built-with { display: none !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }

.plot-panel, .plot-panel > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
}
"""



def wrap_prediction(pred_text, live=False):
    if pred_text in ("—", "", None):
        eyebrow = '<span class="live-dot"></span>Listening…' if live else "Predicted Emotion"
        return f"""<div class="prediction-card">
          <div class="prediction-eyebrow">{eyebrow}</div>
          <div class="prediction-placeholder">awaiting input</div>
        </div>"""
    eyebrow = '<span class="live-dot"></span>Predicted · live' if live else "Predicted Emotion"
    return f"""<div class="prediction-card">
      <div class="prediction-eyebrow">{eyebrow}</div>
      <div class="prediction-value">{pred_text.lower()}</div>
    </div>"""

def wrap_gate(gate_text):
    return f"""<div class="gate-row">
      <span class="gate-label">Modality Gate</span>
      <span class="gate-value">{gate_text}</span>
    </div>"""

def wrap_status(status_text):
    if not status_text:
        return '<div class="status-line">Click Start — the webcam will stay on and predictions update continuously.</div>'
    return f'<div class="status-line">{status_text}</div>'

def wrap_context(text: str) -> str:
    if not text:
        return ""
    return f"""<div style="
        margin-top:16px;
        padding:18px 22px;
        background:#1a1a18;
        border:1px solid #2d2b27;
        border-left:2px solid #a8c5a0;
        border-radius:2px;
        font-family:'Fraunces',serif;
        font-style:italic;
        font-size:1.05em;
        color:#f5f2ea;
        line-height:1.6;
    ">💬 {text}</div>"""


# =========================================================
# UI CALLBACKS
# =========================================================
def get_latest_mirror_frame():
    with _cam_lock:
        if len(FRAME_BUFFER) == 0:
            return None
        _, frame = FRAME_BUFFER[-1]
        frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def tick_inference():
    if not _capture_running.is_set():
        return (
            {label: 0.0 for label in EMOTION_LABELS},
            wrap_prediction("—"),
            wrap_gate("capture not running"),
            wrap_status("press Start to begin"),
            _make_timeline_plot(),
            "",
        )

    if SESSION_START["t"] is None:
        SESSION_START["t"] = time.time()

    clip_path = snapshot_clip_to_mp4()
    if clip_path is None:
        return (
            {label: 0.0 for label in EMOTION_LABELS},
            wrap_prediction("—", live=True),
            wrap_gate("buffering…"),
            wrap_status("filling buffer — first prediction in a moment"),
            _make_timeline_plot(),
            "",
        )

    try:
        probs, pred, gate_desc, status = predict(clip_path)
    finally:
        if os.path.exists(clip_path):
            try: os.remove(clip_path)
            except Exception: pass

    global _pending_response

    if pred not in ("—", "", None) and sum(probs.values()) > 0:
        elapsed = time.time() - SESSION_START["t"]
        HISTORY.append((elapsed, dict(probs)))

        gate_val = _parse_gate_value(gate_desc)
        confidence = probs.get(pred, 0.0)
        tracker.update(pred, confidence, gate_val)

        # ── Write predicted emotion to shared file for bot_walk.py ───────
        try:
            _emotion_state_file = os.path.join(os.path.dirname(__file__), "emotion_state.json")
            with open(_emotion_state_file, "w") as f:
                json.dump({"emotion": pred, "confidence": confidence, "timestamp": time.time()}, f)
        except Exception:
            pass

        response_text = ""
        if _pending_response is not None and _pending_response.done():
            result = _pending_response.result()
            if result:
                response_text = result

        response_type = decide_response_type(tracker)
        if response_type != ResponseType.NONE:
            _pending_response = _groq_executor.submit(
                generate_response, tracker, response_type
            )
    else:
        response_text = ""

    return (
        probs,
        wrap_prediction(pred, live=True),
        wrap_gate(gate_desc),
        wrap_status(status),
        _make_timeline_plot(),
        wrap_context(response_text),
    )

def start_session():
    tracker.history.clear()
    _conversation_history.clear()

    HISTORY.clear()
    SESSION_START["t"] = time.time()
    start_capture_threads()
    return (
        gr.Timer(active=True),
        gr.Timer(active=True),
        wrap_status(f"live · {CLIP_SECONDS:.0f}s rolling window · predictions every {TICK_INTERVAL:.0f}s"),
        _make_timeline_plot(),
    )

def stop_session():
    stop_capture_threads()
    return (
        gr.Timer(active=False),
        gr.Timer(active=False),
        wrap_status("stopped · press Start to resume"),
    )

def clear_history():
    tracker.history.clear()
    _conversation_history.clear()

    HISTORY.clear()
    SESSION_START["t"] = time.time() if _capture_running.is_set() else None
    return (
        {label: 0.0 for label in EMOTION_LABELS},
        wrap_prediction("—", live=_capture_running.is_set()),
        wrap_gate("awaiting input"),
        wrap_status("history cleared"),
        _make_timeline_plot(),
        "",
    )

def analyze_uploaded_video(video_file):
    empty = {label: 0.0 for label in EMOTION_LABELS}
    if video_file is None:
        return (
            empty,
            wrap_prediction("—"),
            wrap_gate("awaiting upload"),
            wrap_status(""),
        )

    probs, pred, gate_desc, status = predict(video_file)

    return (
        probs,
        wrap_prediction(pred, live=False),
        wrap_gate(gate_desc),
        wrap_status(status),
    )


# =========================================================
# UI
# =========================================================
with gr.Blocks(title="Emotion Recognition v4", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:

    gr.HTML(
        """
        <div class="hero">
          <h1 class="hero-title">Emotion <em>recognition</em><br/>in real time.</h1>
          <div class="hero-sub">
            WavLM-Base+ and ViT-Face-Expression with confidence-weighted temporal pooling,
            fused by a learned gate.<br/>
            Use the Live tab for real-time analysis, or upload a pre-recorded video.
          </div>
        </div>
        """
    )

    with gr.Tabs() as tabs:
        # ============================================================
        # TAB 1: LIVE ANALYSIS
        # ============================================================
        with gr.Tab("🎙️ Live Analysis", id="live"):
            mirror_timer = gr.Timer(1.0 / MIRROR_FPS, active=False)
            infer_timer  = gr.Timer(TICK_INTERVAL,    active=False)

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-label">01 · Live Feed</div>')
                    mirror_out = gr.Image(
                        show_label=False,
                        elem_classes="mirror",
                        height=360,
                        type="pil",
                    )
                    with gr.Row(elem_classes="button-row"):
                        start_btn = gr.Button("Start", elem_classes="primary", scale=2)
                        stop_btn  = gr.Button("Stop",  elem_classes="secondary", scale=1)
                        clear_btn = gr.Button("Clear", elem_classes="secondary", scale=1)
                    status_html = gr.HTML(wrap_status(""))

                with gr.Column(scale=4):
                    gr.HTML('<div class="section-label">02 · Current Prediction</div>')
                    pred_html    = gr.HTML(wrap_prediction("—"))
                    gate_html    = gr.HTML(wrap_gate("awaiting input"))
                    context_html = gr.HTML("")
                    gr.HTML('<div class="section-label" style="margin-top:24px;">03 · Class Probabilities</div>')
                    label_out = gr.Label(
                        num_top_classes=4,
                        show_label=False,
                        value={label: 0.0 for label in EMOTION_LABELS},
                    )

            gr.HTML('<div class="section-label" style="margin-top:36px;">04 · Emotion Timeline</div>')
            plot_out = gr.Plot(value=_make_timeline_plot(), show_label=False,
                               elem_classes="plot-panel")

            

            mirror_timer.tick(
                fn=get_latest_mirror_frame,
                inputs=None,
                outputs=[mirror_out],
                show_progress="hidden",
            )

            infer_timer.tick(
                fn=tick_inference,
                inputs=None,
                outputs=[label_out, pred_html, gate_html, status_html, plot_out, context_html],
                show_progress="hidden",
            )

            start_btn.click(
                fn=start_session,
                inputs=None,
                outputs=[mirror_timer, infer_timer, status_html, plot_out],
            )

            stop_btn.click(
                fn=stop_session,
                inputs=None,
                outputs=[mirror_timer, infer_timer, status_html],
            )

            clear_btn.click(
                fn=clear_history,
                inputs=None,
                outputs=[label_out, pred_html, gate_html, status_html, plot_out, context_html],
            )

        # ============================================================
        # TAB 2: UPLOAD VIDEO
        # ============================================================
        with gr.Tab("📁 Upload Video", id="upload"):
            gr.HTML('<div class="section-label">Upload a pre-recorded video for analysis</div>')
            gr.HTML('<div style="font-family: JetBrains Mono, monospace; font-size: 12px; color: #8a857a; margin-bottom: 18px;">'
                    'Upload a video file containing both audio and video. '
                    'The model will run a one-shot multimodal analysis (no live LLM responses).</div>')

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    upload_input = gr.Video(label="Upload Video", sources=["upload"])
                    analyze_btn  = gr.Button("Analyze", elem_classes="primary")
                    upload_status_html = gr.HTML(wrap_status(""))

                with gr.Column(scale=4):
                    gr.HTML('<div class="section-label">Prediction</div>')
                    upload_pred_html = gr.HTML(wrap_prediction("—"))
                    upload_gate_html = gr.HTML(wrap_gate("awaiting upload"))
                    gr.HTML('<div class="section-label" style="margin-top:24px;">Class Probabilities</div>')
                    upload_label_out = gr.Label(
                        num_top_classes=4,
                        show_label=False,
                        value={label: 0.0 for label in EMOTION_LABELS},
                    )

            analyze_btn.click(
                fn=analyze_uploaded_video,
                inputs=[upload_input],
                outputs=[upload_label_out, upload_pred_html, upload_gate_html, upload_status_html],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )