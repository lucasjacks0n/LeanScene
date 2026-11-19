"""Hard negatives from the ClipShots training videos: windows around a model's false alarms.

clipshots.py keeps 4 windows around transitions + 2 random ones per train video, so a model
never trains on its own false alarms on general footage. This re-reads the archive parts, runs
a trained model over each whole train video and saves up to K 64-frame windows centred on its
strongest false alarms, with ground-truth targets (train.json lists every transition, so a
detection with no annotated transition nearby is a false alarm).

    python training/hardneg.py <dir with ClipShots-a, -b, -c> <checkpoint>
    python training/hardneg.py clean <checkpoint>

Only the `train` split is used; the validation videos (clipshots_val_videos.txt) and the test
videos are never read. Output: DATA/clipshots_hardneg/train-<video>.pkl in the format of
clipshots.py, plus done.jsonl (one line per video scanned: frames, false alarms found,
windows kept), which is the resume state: rerunning skips videos listed there.
MAX_VIDEOS (env, default 1200) caps how many videos are scanned. WORKERS (env, default 3) is the
number of scanning processes; each holds a model. LEANSCENE_DEVICE (env) picks the device
(default: CUDA when available, otherwise the CPU).

`clean` writes clean.npz beside the windows: the cleaning step of the paper (see clean()).

Fade false alarms only: HARDNEG_DIR=clipshots_fadeneg FADE_ONLY=1 writes a second set to
DATA/clipshots_fadeneg/, keeping only windows around unannotated *fade* detections
(p fade >= 0.5). Set the same HARDNEG_DIR for `clean`. Read by train.py --clipshots-fadeneg.
"""

import io
import json
import os
import pickle
import random
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clipshots
from clipshots import S

OUT = clipshots.DATA / os.environ.get("HARDNEG_DIR", "clipshots_hardneg")
DEVICE = os.environ.get("LEANSCENE_DEVICE") or None
FADE_ONLY = os.environ.get("FADE_ONLY") == "1"
K, THRESH, NEAR, GAP = 4, 0.5, 4, 16
_model = None


def false_alarms(pr, t):
    """[(peak probability, frame)] of detections with no annotated transition within NEAR frames,
    one per run of detections (runs closer than GAP frames are one event), strongest first."""
    n = min(len(pr), len(t))
    marked = np.convolve((t[:n] != 0).astype(np.float32), np.ones(2 * NEAR + 1), "same") > 0
    score = pr[:n, 2] if FADE_ONLY else pr[:n, 1:].max(1)
    idx = np.flatnonzero((score >= THRESH) & ~marked)
    events, start = [], 0
    for i in range(1, len(idx) + 1):
        if i == len(idx) or idx[i] - idx[i - 1] > GAP:
            run = idx[start:i]
            best = run[np.argmax(score[run])]
            events.append((float(score[best]), int(best)))
            start = i
    return sorted(events, reverse=True)


def scan(path, name, transitions, frame_num, checkpoint):
    """Runs in a worker process (each loads the model once)."""
    global _model
    from leanscene import frame_probs, load_model

    try:
        if _model is None:
            _model = load_model(checkpoint, DEVICE)
        pr = frame_probs(path, _model)
        n = min(int(frame_num), len(pr))
        if n < S:
            return {"video": name, "frames": n, "false_alarms": 0, "windows": 0}
        t = clipshots.targets(n, transitions, True)
        fa = false_alarms(pr, t)
        rng = random.Random(name)
        starts = sorted({min(max(0, f - rng.randint(8, S - 8)), n - S) for _, f in fa[:K]})
        want = {f for s in starts for f in range(s, s + S)}
        frames, i, last = {}, 0, max(want) if want else -1
        cap = cv2.VideoCapture(path)
        while i <= last:
            if i in want:
                ok, f = cap.read()
                if not ok:
                    break
                frames[i] = cv2.imencode(".jpg", cv2.resize(f, (100, 100), interpolation=cv2.INTER_AREA),
                                         [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            elif not cap.grab():
                break
            i += 1
        cap.release()
        out = [([frames[f] for f in range(s, s + S)], t[s : s + S]) for s in starts
               if all(f in frames for f in range(s, s + S))]
        if out:
            with open(OUT / f"train-{name}.pkl", "wb") as fh:
                pickle.dump(out, fh)
        return {"video": name, "frames": n, "false_alarms": len(fa), "windows": len(out)}
    finally:
        os.remove(path)


def main(parts_dir, checkpoint):
    parts = sorted(Path(parts_dir).glob("ClipShots-?"))
    if len(parts) != 3:
        raise SystemExit(f"expected ClipShots-a, -b, -c in {parts_dir}, found {len(parts)}")
    val = clipshots.val_videos()
    ann = json.loads((clipshots.ANNOTATIONS / "train.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "done.jsonl"
    done = {json.loads(l)["video"] for l in log.read_text().splitlines() if l.strip()} if log.exists() else set()
    cap_n = int(os.environ.get("MAX_VIDEOS", 1200))
    tmp = Path(tempfile.mkdtemp(dir=clipshots.DATA))
    pool, pending, seen = ProcessPoolExecutor(int(os.environ.get("WORKERS", 3)), max_tasks_per_child=12), [], len(done)  # workers recycled: on MPS each leaked ~0.3 GB per video

    def collect(block):
        nonlocal pending
        for f in list(pending):
            if block or f.done():
                with open(log, "a") as fh:
                    fh.write(json.dumps(f.result()) + "\n")
                pending.remove(f)
                if not block:
                    continue
                return

    with tarfile.open(fileobj=io.BufferedReader(clipshots.Parts(parts), 1 << 20), mode="r|gz") as tar:
        try:
            for m in tar:
                bits = m.name.split("/")
                if len(bits) < 5 or bits[2] != "videos" or bits[3] != "train" or not m.isfile():
                    continue
                name = bits[4]
                if name not in ann or name in val or name in done:
                    continue
                if seen >= cap_n:
                    break
                p = tmp / name
                with open(p, "wb") as fh:
                    fh.write(tar.extractfile(m).read())
                a = ann[name]
                pending.append(pool.submit(scan, str(p), name, a["transitions"], a["frame_num"], checkpoint))
                seen += 1
                collect(False)
                while len(pending) > 8:  # bound the temp files on disk
                    collect(True)
                if seen % 50 == 0:
                    print(f"{seen}/{cap_n} videos queued", flush=True)
        except (EOFError, tarfile.ReadError) as e:
            print(f"archive ends early ({e}); using what it holds", flush=True)
    while pending:
        collect(True)
    pool.shutdown()
    shutil.rmtree(tmp)
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    fr, fa, w = (sum(r[k] for r in rows) for k in ("frames", "false_alarms", "windows"))
    print(f"done: {len(rows)} videos, {fr} frames, {fa} false alarms ({1000 * fa / max(fr, 1):.2f} per 1k frames), "
          f"{w} windows in {OUT}")


CLEAN_AT, CLEAN_PAD = 0.9, 3


def clean(checkpoint):
    """Many of the strongest "false alarms" are real transitions the annotators missed (of the
    windows whose unannotated peak is >= 0.97, most are plain cuts / fades).
    Writes clean.npz beside the windows: per .pkl, the targets with every unannotated frame the
    model scores >= CLEAN_AT (and CLEAN_PAD frames either side) set to IGNORE, so those frames
    are neither a positive nor a negative. Windows left with no detection in [0.5, CLEAN_AT)
    are flagged unused.   python training/hardneg.py clean <checkpoint>"""
    from leanscene import frame_probs, load_model

    m = load_model(checkpoint, DEVICE)
    out, kept, total, ign = {}, 0, 0, 0
    for p in sorted(OUT.glob("*.pkl")):
        ts = []
        for fr, t in pickle.load(open(p, "rb")):
            x = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in fr]
            sc = np.where(t == 0, frame_probs(x, m)[:, 1:].max(1), 0.0)
            t = t.copy()
            for i in np.flatnonzero(sc >= CLEAN_AT):
                a, e = max(0, i - CLEAN_PAD), min(len(t), i + CLEAN_PAD + 1)
                t[a:e] = np.where(t[a:e] == 0, clipshots.IGNORE, t[a:e])
            use = bool(((sc >= THRESH) & (t == 0)).any())
            total += 1; kept += use; ign += int((sc >= CLEAN_AT).sum())
            ts.append(np.concatenate([[int(use)], t]))
        out[p.name] = np.stack(ts)
    np.savez_compressed(OUT / "clean.npz", **out)
    print(f"{kept} of {total} windows keep a false alarm below {CLEAN_AT}; {ign} frames >= {CLEAN_AT} ignored -> {OUT / 'clean.npz'}")


def load_clean(log=print, src=None):
    """The mined windows with clean() applied: [(jpeg frames, targets)], unused windows dropped."""
    src = Path(src) if src else OUT
    z = np.load(src / "clean.npz")
    out = []
    for p in sorted(src.glob("*.pkl")):
        for (fr, _), row in zip(pickle.load(open(p, "rb")), z[p.name]):
            if row[0]:
                out.append((fr, row[1:].astype(np.int64)))
    log(f"[clipshots] {len(out)} cleaned hard-negative windows from {src.name}")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "clean":
        clean(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2])
