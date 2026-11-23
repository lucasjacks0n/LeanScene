"""LeanScene on ONNX Runtime: the default on the CPU.

The same network as model.LeanScene, exported to the two ONNX graphs the npm package also uses
(weights/features.onnx and weights/head.onnx, written by js/scripts/export_onnx.py). On the CPU
ONNX Runtime runs this small network about twice as fast as PyTorch, with probabilities within
1e-5 of it.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

import cv2
import numpy as np

from .model import FRAME_SIZE, HIST_BINS

try:
    import onnxruntime as _ort
except ImportError:  # no wheel for this Python yet, or not installed: PyTorch is used instead
    _ort = None

PIXELS = FRAME_SIZE * FRAME_SIZE


def available() -> bool:
    """Whether ONNX Runtime can be imported."""
    return _ort is not None


def color_hist(frames: np.ndarray) -> np.ndarray:
    """(n,H,W,3) uint8 BGR -> (n, HIST_BINS**3) float32, equal to model.color_hist.

    The model's bin of a byte v is floor(v/255 * (bins - 0.001)), which for 4 bins is v >> 6:
    four equal bins over [0, 256), what cv2.calcHist counts (in C++, without holding the GIL).
    """
    assert HIST_BINS == 4
    out = np.empty((len(frames), HIST_BINS ** 3), np.float32)
    for i, f in enumerate(frames):
        out[i] = cv2.calcHist([f], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]).reshape(-1)
    out /= np.float32(frames.shape[1] * frames.shape[2])
    return out


class OnnxModel:
    """The bundled weights on ONNX Runtime's CPU provider. Use load_model() to get one."""

    def __init__(self, threads: Optional[int] = None):
        if _ort is None:
            raise ImportError("onnxruntime is not installed; pip install onnxruntime")
        weights = resources.files("leanscene") / "weights"
        self.meta = json.loads((weights / "meta.json").read_text())
        if self.meta["size"] != FRAME_SIZE or self.meta["histBins"] != HIST_BINS:
            raise ValueError("weights/meta.json does not match this version of leanscene")
        so = _ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        # Idle worker threads sleep instead of spinning. Spinning took the cores the video decoder
        # needs: end to end it cost about a quarter of the speed (measured on an Apple M1 Pro).
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self._features = _ort.InferenceSession((weights / "features.onnx").read_bytes(), so, providers=providers)
        self._head = _ort.InferenceSession((weights / "head.onnx").read_bytes(), so, providers=providers)
        self.dim, self.feat, self.receptive_field = self.meta["dim"], self.meta["feat"], self.meta["receptiveField"]
        self.checkpoint = self.meta["checkpoint"]

    def frame_features(self, frames: np.ndarray, hist: Optional[np.ndarray] = None) -> np.ndarray:
        """(n,100,100,3) uint8 BGR -> (n, dim) float32, as LeanScene.frame_features."""
        x = np.ascontiguousarray(frames.transpose(0, 3, 1, 2)).astype(np.float32)
        x *= np.float32(1 / 255)
        y = self._features.run(None, {"x": x})[0]  # [CNN mean | thumbnail | grid]; the histogram goes after the mean
        nb = HIST_BINS ** 3
        out = np.empty((len(frames), self.dim), np.float32)
        out[:, : self.feat] = y[:, : self.feat]
        out[:, self.feat : self.feat + nb] = color_hist(frames) if hist is None else hist
        out[:, self.feat + nb :] = y[:, self.feat :]
        return out

    def head_probs(self, features: np.ndarray) -> np.ndarray:
        """(S, dim) features of consecutive frames -> (S, 3) probabilities."""
        return self._head.run(None, {"x": features[None]})[0]

    def __repr__(self) -> str:
        return f"OnnxModel({self.checkpoint!r}, onnxruntime {_ort.__version__})"
