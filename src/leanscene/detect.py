"""Run LeanScene over a video: per-frame probabilities, the decision rule, shots."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Callable, Iterable, Iterator, Optional, Union

import cv2
import numpy as np
import torch

from .model import FRAME_SIZE, LeanScene

# The decision rule used for every number in the paper. It was fixed before the test sets
# were scored.
CUT_THRESHOLD = 0.7
FADE_THRESHOLD = 0.8
FADE_CLOSE = 8  # fade frames separated by at most this many frames are joined into one run

MODELS = {"leanscene": "leanscene.pt"}  # bundled weights: the paper's model

Source = Union[str, "os.PathLike[str]", Iterable[np.ndarray]]


def default_device() -> torch.device:
    """CUDA when available, otherwise the CPU.

    Apple's MPS backend is faster but is not chosen automatically: in our runs a process that
    fed it many different input lengths kept growing in memory. Pass device="mps" to use it.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name: Union[str, "os.PathLike[str]"] = "leanscene", device=None) -> LeanScene:
    """Load bundled weights by name (see MODELS) or a checkpoint file by path."""
    if str(name) in MODELS:
        with resources.as_file(resources.files("leanscene") / "weights" / MODELS[str(name)]) as path:
            ck = torch.load(path, map_location="cpu", weights_only=True)
    elif os.path.exists(name):
        ck = torch.load(name, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unknown model {str(name)!r}: expected one of {sorted(MODELS)} or a checkpoint path")
    model = LeanScene(**ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(torch.device(device) if device is not None else default_device()).eval()


def _frames(source: Source) -> Iterator[np.ndarray]:
    """Yield every frame as FRAME_SIZE x FRAME_SIZE BGR uint8."""
    size = (FRAME_SIZE, FRAME_SIZE)
    if isinstance(source, (str, os.PathLike)):
        if not os.path.exists(source):
            raise FileNotFoundError(str(source))
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"cannot open {source} as a video")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.resize(frame, size)
        finally:
            cap.release()
    else:
        for frame in source:
            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError(f"frames must be HxWx3 uint8 in BGR order, got {frame.shape} {frame.dtype}")
            yield frame if frame.shape[:2] == size else cv2.resize(frame, size)


@torch.inference_mode()
def frame_probs(source: Source, model: Optional[LeanScene] = None, device=None, batch: Optional[int] = None,
                window: int = 4096, progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
    """Per-frame probabilities of same / cut / fade, as a float32 array of shape (frames, 3).

    source: a video path, or an iterable of HxWx3 uint8 frames in BGR order (OpenCV's).
    The video is streamed: frames are encoded in batches of `batch` and the temporal head runs
    over `window` frames at a time with more context on each side than its receptive field, so
    memory stays flat on long videos and the result equals a single pass up to float rounding.
    progress: called with the number of frames read so far, after each batch.
    """
    if model is None:
        model = load_model(device=device)
    elif device is not None:
        model = model.to(device)
    model.eval()
    dev = next(model.parameters()).device
    if batch is None:
        # Measured on an Apple M1 Pro: small batches are about twice as fast on the CPU.
        batch = 8 if dev.type == "cpu" else 512
    ctx = max(64, model.receptive_field // 2 + 1)

    out: list[np.ndarray] = []
    chunks: list[torch.Tensor] = []  # features not yet labelled; the first `done` rows are left context
    held, done, seen = 0, 0, 0

    def encode(frames: list[np.ndarray]) -> None:
        nonlocal held
        x = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).float().div_(255)
        chunks.append(model.frame_features(x))
        held += len(frames)
        if progress is not None:
            nonlocal seen
            seen += len(frames)
            progress(seen)

    def label(final: bool) -> None:
        nonlocal chunks, held, done
        hi = held if final else held - ctx
        if hi <= done:
            return
        f = torch.cat(chunks)
        out.append(torch.softmax(model.head(f[None])[0], 1)[done:hi].float().cpu().numpy())
        keep = max(0, hi - ctx)
        chunks, held, done = [f[keep:]], held - keep, hi - keep

    pending: list[np.ndarray] = []
    for frame in _frames(source):
        pending.append(frame)
        if len(pending) == batch:
            encode(pending)
            pending = []
            if held - done >= window + ctx:
                label(False)
    if pending:
        encode(pending)
    label(True)
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def label_frames(probs: np.ndarray, cut: float = CUT_THRESHOLD, fade: float = FADE_THRESHOLD,
                 close: int = FADE_CLOSE) -> np.ndarray:
    """Apply the decision rule: (frames, 3) probabilities -> labels 0 same, 1 cut, 2 fade."""
    probs = np.asarray(probs)
    labels = np.zeros(len(probs), np.int64)
    labels[probs[:, 2] >= fade] = 2
    labels[probs[:, 1] >= cut] = 1
    fades = np.flatnonzero(labels == 2)
    for a, b in zip(fades[:-1], fades[1:]):
        if 1 < b - a <= close + 1:
            labels[a:b] = 2
    return labels


@dataclass(frozen=True)
class Transition:
    """A maximal run of frames not labelled same. start and end are inclusive frame indices.

    kind is "cut" when the run holds no fade frame, else "gradual". For a cut, `end` is the
    first frame of the new shot; start < end only when a few junk frames sit between two shots.
    """

    start: int
    end: int
    kind: str


def find_transitions(labels: np.ndarray) -> list[Transition]:
    labels = np.asarray(labels)
    on = np.concatenate([[0], (labels != 0).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(on))
    return [Transition(int(s), int(e) - 1, "gradual" if (labels[s:e] == 2).any() else "cut")
            for s, e in zip(edges[::2], edges[1::2])]


def find_shots(transitions: Iterable[Transition], frames: int) -> list[tuple[int, int]]:
    """Shots as inclusive (first frame, last frame) pairs.

    A cut starts the new shot on its last frame. The frames of a gradual transition belong to
    neither neighbour.
    """
    shots, start = [], 0
    for t in transitions:
        if t.start - 1 >= start:
            shots.append((start, t.start - 1))
        start = t.end if t.kind == "cut" else t.end + 1
    if start <= frames - 1:
        shots.append((start, frames - 1))
    return shots


@dataclass
class Result:
    frames: int
    fps: Optional[float]  # None when the source is not a file or the file does not say
    transitions: list[Transition]
    shots: list[tuple[int, int]]
    probs: np.ndarray = field(repr=False)  # (frames, 3): same, cut, fade

    def to_dict(self) -> dict:
        """JSON-ready. Times are in seconds and present only when fps is known; a shot's
        end time is the end of its last frame."""
        def t(frame: int):
            return round(frame / self.fps, 4)

        shots = [{"start": a, "end": b} for a, b in self.shots]
        transitions = [{"start": x.start, "end": x.end, "kind": x.kind} for x in self.transitions]
        if self.fps:
            for d in shots + transitions:
                d["start_time"], d["end_time"] = t(d["start"]), t(d["end"] + 1)
        return {"frames": self.frames, "fps": self.fps, "shots": shots, "transitions": transitions}


def _fps(path) -> Optional[float]:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return float(fps) if fps and fps > 0 else None


def detect(source: Source, model: Optional[LeanScene] = None, device=None, cut: float = CUT_THRESHOLD,
           fade: float = FADE_THRESHOLD, close: int = FADE_CLOSE, batch: Optional[int] = None,
           progress: Optional[Callable[[int], None]] = None) -> Result:
    """Find the shot transitions of a video.

    source: a video path, or an iterable of HxWx3 uint8 BGR frames.
    model: a loaded model to reuse across videos (load_model()); loaded on each call otherwise.
    cut, fade: probability thresholds. Raise them for fewer false alarms, lower them to miss less.
    progress: called with the number of frames read so far, after each batch.
    """
    probs = frame_probs(source, model=model, device=device, batch=batch, progress=progress)
    transitions = find_transitions(label_frames(probs, cut, fade, close))
    fps = _fps(source) if isinstance(source, (str, os.PathLike)) else None
    return Result(len(probs), fps, transitions, find_shots(transitions, len(probs)), probs)
