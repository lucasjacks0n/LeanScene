"""Score LeanScene on BBC Planet Earth, RAI or ClipShots with the paper's protocol.

    python eval/evaluate.py rai /data/rai
    python eval/evaluate.py bbc /data/bbc
    python eval/evaluate.py clipshots /data/clipshots/videos/test /data/clipshots/annotations/test.json

The metric is TransNet V2's: F1 over transitions with a two-frame tolerance, counts summed
over the videos of a set. See eval/README.md for the dataset layouts and the expected numbers.

predictions_to_scenes and evaluate_scenes follow training/metrics_utils.py of
https://github.com/soCzech/TransNetV2 (MIT licence, Copyright (c) 2020 Tomáš Souček).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import leanscene


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    """Binary per-frame transition labels -> shots as [first frame, last frame] rows."""
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes: np.ndarray, pred_scenes: np.ndarray, n_frames_miss_tolerance: int = 2):
    """(true positives, false positives, false negatives) over transitions."""
    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)
    i = j = tp = fp = fn = 0
    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1; i += 1
        elif i == len(gt_trans):
            fp += 1; j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1; j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1; i += 1
        else:
            i += 1; j += 1; tp += 1
    return tp, fp, fn


def f1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def bbc(root: str):
    """bbc_<NN>.mp4 anywhere under root, annotations/shots/<NN>_*.txt (one shot per line)."""
    for path in sorted(glob.glob(os.path.join(root, "**", "bbc_*.mp4"), recursive=True)):
        ep = os.path.basename(path).split("_")[1].split(".")[0]
        ann = glob.glob(os.path.join(root, "annotations", "shots", f"{ep}_*"))[0]
        shots = np.loadtxt(ann, dtype=np.int32, ndmin=2) + 1  # TransNet V2's consolidate_datasets.py convention
        if shots[0][0] == 1:
            shots[0][0] = 0
        yield ep, path, shots


def rai(root: str):
    """<id>.mp4 and <id>_gt.txt: one shot per line, first and last frame, 0-based."""
    for path in sorted(glob.glob(os.path.join(root, "*.mp4"))):
        yield os.path.basename(path)[:-4], path, np.loadtxt(path[:-4] + "_gt.txt", dtype=np.int32, ndmin=2)


def clipshots(videos: str, annotations: str):
    """The test videos and ClipShots' annotations/test.json."""
    ann = json.load(open(annotations))
    for path in sorted(glob.glob(os.path.join(videos, "*.mp4"))):
        a = ann.get(os.path.basename(path))
        if a is None:
            continue
        shots, start = [], 0
        for x, y in a["transitions"]:
            shots.append([start, int(x)])
            start = int(y)
        shots.append([start, int(a["frame_num"]) - 1])
        yield os.path.basename(path), path, np.array(shots, np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", choices=["bbc", "rai", "clipshots"])
    ap.add_argument("paths", nargs="+", help="dataset root (bbc, rai); videos folder and test.json (clipshots)")
    ap.add_argument("--model", default="leanscene")
    ap.add_argument("--device")
    ap.add_argument("--cut", type=float, default=leanscene.CUT_THRESHOLD)
    ap.add_argument("--fade", type=float, default=leanscene.FADE_THRESHOLD)
    ap.add_argument("-v", "--verbose", action="store_true", help="print every video")
    args = ap.parse_args()
    if len(args.paths) != (2 if args.dataset == "clipshots" else 1):
        ap.error("clipshots takes the videos folder and test.json; bbc and rai take the dataset root")

    model = leanscene.load_model(args.model, args.device)
    total, videos, frames = (0, 0, 0), 0, 0
    for name, path, gt in {"bbc": bbc, "rai": rai, "clipshots": clipshots}[args.dataset](*args.paths):
        probs = leanscene.frame_probs(path, model=model)
        if len(probs) < 64:  # the paper skips files that do not decode
            print(f"skipped (unreadable): {name}", flush=True)
            continue
        labels = leanscene.label_frames(probs, args.cut, args.fade)
        r = evaluate_scenes(gt, predictions_to_scenes((labels != 0).astype(np.uint8)))
        total = tuple(a + b for a, b in zip(total, r))
        videos, frames = videos + 1, frames + len(probs)
        if args.verbose:
            print(f"{name}: {len(probs)} frames, {len(gt) - 1} transitions, F1 {f1(*r)[0] * 100:.1f}, tp/fp/fn {r}", flush=True)
    if not videos:
        raise SystemExit(f"no {args.dataset} videos found under {args.paths[0]}")
    score, p, r = f1(*total)
    print(f"{args.dataset}: {videos} videos, {frames} frames, tp/fp/fn {total}")
    print(f"F1 {score * 100:.2f}  precision {p * 100:.2f}  recall {r * 100:.2f}")


if __name__ == "__main__":
    main()
