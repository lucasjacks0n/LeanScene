"""Training windows around the hand labels in labels/labels.csv.

train.py --real mixes them into the synthetic batches. A label is a video id and a frame
index; the videos are not distributed, so you download them yourself (labels/README.md) into
one directory as <video_id>.mp4 and pass it as --videos.

Only frames near a label are supervised; everything else in the window is ignored
(-100), since nobody checked it for other transitions:
  cut / fade at f: f is the label, f±1 ignored (the cut may be a frame off),
                   f±2..f±8 count as "same" (a shot boundary is rarely closer)
  same at f:       f-3..f+3 count as "same"

    python training/labels.py <dir with the videos>     # read the frames and print what was found

Frames are cached as one .npz per window in DATA/label_frames (see training/README.md), so a
video is decoded once.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clipshots

LABELS = Path(__file__).resolve().parent.parent / "labels" / "labels.csv"
CACHE_DIR = clipshots.DATA / "label_frames"  # frames only; labels are always read fresh
HALF = 40  # frames kept on each side of a label
NEAR = 8   # how far a cut/fade label vouches for "same" around it
IGNORE = -100
CLASS = {"same": 0, "cut": 1, "fade": 2}


def _cache_path(video_id, start, n, size):
    return CACHE_DIR / f"{video_id}-{start}-{n}-{size[0]}x{size[1]}.npz"


def _read_windows(path, starts, n, size):
    """{start: [frames]} for each start, from one pass over the video. Frames are counted by
    reading the file frame by frame, which is what the label indices refer to."""
    want = {f for s in starts for f in range(s, s + n)}
    frames, i, last = {}, 0, max(want)
    cap = cv2.VideoCapture(str(path))
    while i <= last:
        if i in want:
            ok, f = cap.read()
            if not ok:
                break
            frames[i] = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        elif not cap.grab():
            break
        i += 1
    cap.release()
    return {s: [frames[f] for f in range(s, s + n) if f in frames] for s in starts}


def targets(frame0, n, labels):
    """Per-frame targets for frames frame0..frame0+n-1 given (frame, label) pairs."""
    t = np.full(n, IGNORE, np.int64)
    for f, lab in labels:  # "same" zones first, so exact labels and ignore rings win
        lo, hi = (f - 3, f + 3) if lab == "same" else (f - NEAR, f + NEAR)
        for g in range(max(lo, frame0), min(hi, frame0 + n - 1) + 1):
            t[g - frame0] = 0
    for f, lab in labels:
        if lab != "same":
            for g in (f - 1, f + 1):
                if frame0 <= g < frame0 + n:
                    t[g - frame0] = IGNORE
    for f, lab in labels:
        if frame0 <= f < frame0 + n:
            t[f - frame0] = CLASS[lab]
    return t


def load_windows(videos_dir=None, size=(100, 100), log=print, labels_file=None):
    """[(frames, targets)] for every label whose frames are cached or whose video is in
    videos_dir. Labels on a missing video are skipped and counted in the log."""
    with open(labels_file or LABELS, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append((int(r["frame"]), r["label"]))
    n = 2 * HALF + 1
    out, missing = [], []
    for vid, labs in sorted(by_video.items()):
        starts = [max(0, f - HALF) for f, _ in labs]
        todo = sorted({s for s in starts if not _cache_path(vid, s, n, size).exists()})
        if todo:
            video = Path(videos_dir) / f"{vid}.mp4" if videos_dir else None
            if video is None or not video.exists():
                missing.append(vid)
            else:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                for s, fr in _read_windows(video, todo, n, size).items():
                    if fr:
                        np.savez_compressed(_cache_path(vid, s, n, size), frames=np.stack(fr))
        for s in starts:
            p = _cache_path(vid, s, n, size)
            if not p.exists():
                continue
            fr = list(np.load(p)["frames"])
            if len(fr) < 64:  # batches need full 64-frame sequences
                continue
            out.append((fr, targets(s, len(fr), labs)))
    if missing:
        where = f"not found in {videos_dir}" if videos_dir else "have uncached frames and no --videos directory was given"
        log(f"[labels] {len(missing)} of {len(by_video)} videos {where}; their uncached labels are skipped "
            f"(first: {missing[:5]})")
    return out


def sample(windows, rng, S=64):
    """A random S-frame slice of a random labeled window."""
    fr, t = windows[rng.randrange(len(windows))]
    s = rng.randint(0, len(fr) - S)
    return list(fr[s : s + S]), t[s : s + S].copy()


if __name__ == "__main__":
    w = load_windows(sys.argv[1] if len(sys.argv) > 1 else None)
    n = {k: sum(int((t == v).sum()) for _, t in w) for k, v in CLASS.items()}
    print(f"{len(w)} labeled windows; supervised frames: {n}")
