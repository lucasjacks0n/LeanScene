"""Command line: leanscene VIDEO [VIDEO ...]"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

from . import __version__
from .detect import CUT_THRESHOLD, FADE_THRESHOLD, detect, load_model


def _clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _short(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _Progress:
    """One status line on standard error, redrawn in place a few times a second."""

    def __init__(self, path: str, label: str):
        cap = cv2.VideoCapture(path)
        # The container's frame count: usually right, sometimes off by a few, sometimes absent.
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.label, self.start, self.last, self.width = label, time.monotonic(), 0.0, 0

    def _write(self, text: str, end: str = "") -> None:
        pad = " " * max(0, self.width - len(text))
        self.width = len(text)
        sys.stderr.write(f"\r{text}{pad}{end}")
        sys.stderr.flush()

    def __call__(self, frames: int) -> None:
        now = time.monotonic()
        if now - self.last < 0.2:
            return
        self.last = now
        rate = frames / max(now - self.start, 1e-6)
        if self.total >= frames:
            left = _short((self.total - frames) / rate) if rate > 0 else "?"
            text = (f"{self.label}: {frames:,} / {self.total:,} frames ({100 * frames / self.total:.0f}%), "
                    f"{rate:,.0f} frames/s, {left} left")
        else:
            text = f"{self.label}: {frames:,} frames, {rate:,.0f} frames/s"
        self._write(text)

    def finish(self, frames: int) -> None:
        took = time.monotonic() - self.start
        self._write(f"{self.label}: {frames:,} frames in {_short(took)} ({frames / max(took, 1e-6):,.0f} frames/s)", "\n")

    def clear(self) -> None:
        self._write("")
        sys.stderr.write("\r")


def _table(path: str, d: dict, out) -> None:
    timed = d["fps"] is not None
    fps = f"{d['fps']:.3f} fps" if timed else "fps unknown"
    print(f"{path}: {d['frames']} frames, {fps}, {len(d['shots'])} shots, {len(d['transitions'])} transitions", file=out)
    print(f"{'shot':>5} {'start':>8} {'end':>8}" + (f"  {'start time':>12} {'end time':>12}" if timed else ""), file=out)
    for i, s in enumerate(d["shots"], 1):
        times = f"  {_clock(s['start_time']):>12} {_clock(s['end_time']):>12}" if timed else ""
        print(f"{i:>5} {s['start']:>8} {s['end']:>8}{times}", file=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="leanscene", description="Detect shot transitions (cuts and gradual transitions) in videos.")
    ap.add_argument("videos", nargs="+", help="video files")
    ap.add_argument("-f", "--format", choices=["table", "json", "csv"], default="table",
                    help="table: one row per shot (default). json: shots and transitions. csv: one row per shot")
    ap.add_argument("-o", "--output", help="write to this file instead of standard output")
    ap.add_argument("--model", default="leanscene", help="a checkpoint path, to use weights other than the bundled ones")
    ap.add_argument("--device", help="cpu, cuda or mps (default: cuda when available, else cpu)")
    ap.add_argument("--cut", type=float, default=CUT_THRESHOLD, help=f"cut threshold (default {CUT_THRESHOLD})")
    ap.add_argument("--fade", type=float, default=FADE_THRESHOLD, help=f"gradual-transition threshold (default {FADE_THRESHOLD})")
    ap.add_argument("--probs", metavar="FILE.npy",
                    help="also save the per-frame probabilities (frames x 3: same, cut, fade); one video only")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="no progress line (it goes to standard error, and is shown only in a terminal)")
    ap.add_argument("--version", action="version", version=f"leanscene {__version__}")
    args = ap.parse_args(argv)
    if args.probs and len(args.videos) > 1:
        ap.error("--probs takes a single video")

    show = not args.quiet and sys.stderr.isatty()
    if show:
        sys.stderr.write("Loading model...")
        sys.stderr.flush()
    try:
        model = load_model(args.model, args.device)
    except ValueError as e:
        ap.error(str(e))
    if show:
        sys.stderr.write("\r" + " " * 16 + "\r")
    results, failed = {}, False
    for n, path in enumerate(args.videos, 1):
        bar = None
        try:
            if show and os.path.exists(path):
                bar = _Progress(path, os.path.basename(path) + (f" [{n}/{len(args.videos)}]" if len(args.videos) > 1 else ""))
            r = detect(path, model=model, cut=args.cut, fade=args.fade, progress=bar)
            if bar and r.frames:
                bar.finish(r.frames)
            elif bar:
                bar.clear()
        except (FileNotFoundError, ValueError) as e:
            if bar:
                bar.clear()
            print(f"leanscene: {path}: {e if isinstance(e, ValueError) else 'no such file'}", file=sys.stderr)
            failed = True
            continue
        if r.frames == 0:
            print(f"leanscene: {path}: no frames could be decoded", file=sys.stderr)
            failed = True
            continue
        results[path] = r.to_dict()
        if args.probs:
            np.save(args.probs, r.probs)

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        if args.format == "json":
            json.dump(results if len(args.videos) > 1 else next(iter(results.values()), {}), out, indent=1)
            out.write("\n")
        elif args.format == "csv":
            w = csv.writer(out)
            w.writerow(["video", "shot", "start_frame", "end_frame", "start_time", "end_time"])
            for path, d in results.items():
                for i, s in enumerate(d["shots"], 1):
                    w.writerow([path, i, s["start"], s["end"], s.get("start_time", ""), s.get("end_time", "")])
        else:
            for n, (path, d) in enumerate(results.items()):
                if n:
                    print(file=out)
                _table(path, d, out)
    finally:
        if args.output:
            out.close()
    return 1 if failed else 0
