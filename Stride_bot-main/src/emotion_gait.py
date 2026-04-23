import numpy as np
import math


def get_emotion_params(emotion: str, t: float = 0.0) -> dict:
    """Return gait parameters for the given emotion.

    Parameters
    ----------
    emotion : str
        One of "happy", "sad", "neutral", "angry".
    t : float
        Current simulation time in seconds (used for time-varying oscillations).

    Returns
    -------
    dict with keys: L, angle, Lrot, T, sda, offset, pos, orn, bodytoFeet_override
    """

    if emotion == "happy":
        phase = 2 * math.pi * t / 1.0
        pitch = -0.10 + 0.15 * math.sin(phase)           # bouncy nod
        yaw   = 0.25 * math.sin(2 * math.pi * t / 0.6)   # tail-wag
        roll  = 0.10 * math.sin(2 * math.pi * t / 0.6)
        return {
            "L":      0.12,
            "angle":  0,
            "Lrot":   0,
            "T":      0.32,
            "sda":    0,
            "offset": [0.5, 0.5, 0.0, 0.0],
            "pos":    np.array([0.0, 0.0, 0.0]),
            "orn":    np.array([roll, pitch, yaw]),
            "bodytoFeet_override": None,
        }

    elif emotion == "sad":
        # Sadness via hunched pitch + slow tiny steps — NO foot geometry change
        pitch = 0.15 + 0.03 * math.sin(2 * math.pi * t / 4.0)
        return {
            "L":      0.04,           # barely moving
            "angle":  0,
            "Lrot":   0,
            "T":      0.8,            # slow steps
            "sda":    0,
            "offset": [0.5, 0.5, 0.0, 0.0],
            "pos":    np.array([0.0, 0.0, 0.0]),     # no body height change
            "orn":    np.array([0.0, 0.05+pitch, 0.0]),    # head drooped forward
            "bodytoFeet_override": None,              # same foot geometry as default
        }

    elif emotion == "angry":
        return {
            "L":      0.5,
            "angle":  0,
            "Lrot":   0.10,
            "T":      0.40,
            "sda":    0,
            "offset": [0.5, 0.5, 0.0, 0.0],
            "pos":    np.array([0.0, 0.0, 0.0]),
            "orn":    np.array([0.0, -0.03, 0.0]),
            "bodytoFeet_override": None,
        }

    elif emotion == "neutral":
        y_sway = 0.02 * math.sin(2 * math.pi * t / 0.8)
        return {
            "L":      0.0,
            "angle":  0,
            "Lrot":   0,
            "T":      0.40,
            "sda":    0,
            "offset": [0.5, 0.5, 0.0, 0.0],
            "pos":    np.array([0.0, y_sway, 0.0]),
            "orn":    np.array([0.0, 0.0, 0.0]),
            "bodytoFeet_override": None,
        }