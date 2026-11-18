"""LeanScene: a 0.39M-parameter network for fast shot transition detection."""

from .detect import (CUT_THRESHOLD, FADE_CLOSE, FADE_THRESHOLD, MODELS, Result, Transition, default_device, detect,
                     find_shots, find_transitions, frame_probs, label_frames, load_model)
from .model import LeanScene

__version__ = "0.1.2"

__all__ = ["CUT_THRESHOLD", "FADE_CLOSE", "FADE_THRESHOLD", "MODELS", "LeanScene", "Result", "Transition",
           "default_device", "detect", "find_shots", "find_transitions", "frame_probs", "label_frames", "load_model"]
