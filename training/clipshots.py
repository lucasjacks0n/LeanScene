"""ClipShots (https://github.com/Tangshitao/ClipShots) as training windows.

Reads the split data.tar.gz parts in order (a truncated set of parts still yields every
video it fully contains), keeps test videos as .mp4 for evaluation, and turns train and
only_gradual videos into 64-frame windows at 100x100, JPEG-encoded, one .pkl per video.
Source videos are deleted as soon as they are processed, so disk use stays small.

Targets per frame: transitions [a, b] are (last frame of the old shot, first frame of
the new one). b - a == 1 is a cut: b is 1, b±1 ignored. Longer ones are gradual: frames
a+1..b-1 are 2 (fade). train.json lists every transition, so all other frames are 0.
only_gradual.json lists gradual transitions only (cuts unlabeled), so there only the
fades and 3 frames either side are supervised.

    python training/clipshots.py <dir with ClipShots-a, -b, -c>
    python training/clipshots.py --val <dir with ClipShots-a, -b, -c>

The 906 train videos named in clipshots_val_videos.txt are the validation set: they never
become training windows, and --val extracts them as .mp4. Rerunning skips videos whose
.pkl / .mp4 exists. Paths and sizes: training/README.md.
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

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LEANSCENE_DATA", HERE / "data"))  # everything this directory's scripts write
ANNOTATIONS = DATA / "ClipShots" / "annotations"  # DATA/ClipShots: a copy of the ClipShots repository
OUT = DATA / "clipshots_windows"
TEST = DATA / "clipshots_test"
VAL = DATA / "clipshots_val"
S, IGNORE = 64, -100
PER_VIDEO = {"train": 6, "only_gradual": 4}  # windows kept per video


class Parts(io.RawIOBase):
    """The parts as one stream; a missing tail ends the stream early."""

    def __init__(self, paths):
        self.paths, self.f = list(paths), None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.f is None:
                if not self.paths:
                    return 0
                self.f = open(self.paths.pop(0), "rb")
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.f = None


def targets(n, transitions, exhaustive):
    t = np.zeros(n, np.int64) if exhaustive else np.full(n, IGNORE, np.int64)
    for a, b in transitions:
        a, b = int(a), int(b)
        if b - a == 1:
            for g in (b - 1, b + 1):
                if 0 <= g < n:
                    t[g] = IGNORE
            if b < n:
                t[b] = 1
        else:
            if not exhaustive:
                for g in list(range(a - 2, a + 1)) + list(range(b, b + 3)):
                    if 0 <= g < n:
                        t[g] = 0
            t[max(0, a + 1) : min(n, b)] = 2
    return t


def windows_for(n, transitions, k, rng, exhaustive):
    """Start frames: mostly windows holding a random transition, the rest anywhere."""
    starts = []
    centres = [int(b) for _, b in transitions if int(b) < n]
    rng.shuffle(centres)
    for c in centres[: k if not exhaustive else k - 2]:
        starts.append(min(max(0, c - rng.randint(8, S - 8)), n - S))
    while exhaustive and len(starts) < k:
        starts.append(rng.randint(0, n - S))
    return sorted(set(s for s in starts if s >= 0))


def process(path, name, split, transitions, frame_num):
    rng = random.Random(name)
    try:
        cap = cv2.VideoCapture(path)
        n = int(frame_num)
        exhaustive = split == "train"
        if n < S:
            return name, 0
        starts = windows_for(n, transitions, PER_VIDEO[split], rng, exhaustive)
        want = {f for s in starts for f in range(s, s + S)}
        frames, i, last = {}, 0, max(want) if want else -1
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
        t = targets(n, transitions, exhaustive)
        out = [([frames[f] for f in range(s, s + S)], t[s : s + S]) for s in starts
               if all(f in frames for f in range(s, s + S))]
        if out:
            with open(OUT / f"{split}-{name}.pkl", "wb") as fh:
                pickle.dump(out, fh)
        return name, len(out)
    finally:
        os.remove(path)


def val_videos():
    """Names of the ClipShots train videos held out as the validation set."""
    return set((HERE / "clipshots_val_videos.txt").read_text().split())


def main(parts_dir):
    parts = sorted(Path(parts_dir).glob("ClipShots-?"))  # the parts may sit elsewhere, e.g. an external disk
    val = val_videos()
    ann = {}
    for split in ("train", "only_gradual", "test"):
        ann[split] = json.loads((ANNOTATIONS / f"{split}.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    TEST.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=DATA))
    done = {p.stem.split("-", 1)[1] for p in OUT.glob("*.pkl")}
    pool, pending, counts = ProcessPoolExecutor(4), [], {"train": 0, "only_gradual": 0, "test": 0}
    with tarfile.open(fileobj=io.BufferedReader(Parts(parts), 1 << 20), mode="r|gz") as tar:
        try:
            for m in tar:
                bits = m.name.split("/")
                if len(bits) < 5 or bits[2] != "videos" or not m.isfile():
                    continue
                split, name = bits[3], bits[4]
                if split not in ann or name not in ann[split]:
                    continue
                if split == "test":
                    dst = TEST / name
                    if not dst.exists():
                        with open(str(dst) + ".part", "wb") as fh:
                            fh.write(tar.extractfile(m).read())
                        os.replace(str(dst) + ".part", dst)
                    counts["test"] += 1
                    continue
                if name in done or (split == "train" and name in val):
                    continue
                p = tmp / name
                with open(p, "wb") as fh:
                    fh.write(tar.extractfile(m).read())
                a = ann[split][name]
                pending.append(pool.submit(process, str(p), name, split, a["transitions"], a["frame_num"]))
                counts[split] += 1
                while sum(not f.done() for f in pending) > 16:  # bound the temp files on disk
                    pending[0].result()
                    pending = [f for f in pending if not f.done()]
                if sum(counts.values()) % 200 == 0:
                    print(counts, flush=True)
        except (EOFError, tarfile.ReadError) as e:  # missing later parts
            print(f"archive ends early ({e}); using what it holds", flush=True)
    for f in pending:
        f.result()
    pool.shutdown()
    shutil.rmtree(tmp)  # may hold the truncated last video
    n = sum(len(pickle.load(open(p, "rb"))) for p in OUT.glob("*.pkl"))
    print(f"done: {counts}; {n} windows in {OUT}")


class Windows:
    """The windows of a directory of .pkl files as a read-only list of (jpeg frames, targets).
    Only the targets stay in memory; frames are read from the window's file when it is asked
    for (the 19,506 windows are 5 GB of JPEG, which pushed training into swap on a 32 GB Mac)."""

    def __init__(self, refs, targets):
        self.refs, self.targets, self._last = refs, targets, (None, None)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        path, k = self.refs[i]
        if self._last[0] != path:
            self._last = (path, pickle.load(open(path, "rb")))
        return self._last[1][k][0], self.targets[i]

    def where(self, keep):
        """The windows whose targets satisfy keep(targets)."""
        idx = [i for i, t in enumerate(self.targets) if keep(t)]
        return Windows([self.refs[i] for i in idx], [self.targets[i] for i in idx])


def load_windows(log=print, src=None):
    """Windows (list-like: [(jpeg frames, targets)]) over every .pkl; frames decode in sample().
    src: another directory in the same format (hardneg.py's output)."""
    refs, targets = [], []
    for p in sorted((src or OUT).glob("*.pkl")):
        for k, (_, t) in enumerate(pickle.load(open(p, "rb"))):
            refs.append((p, k))
            targets.append(np.asarray(t))
    log(f"[clipshots] {len(refs)} windows" + (f" from {src}" if src else ""))
    return Windows(refs, targets)


def sample(windows, rng, S=S):
    fr, t = windows[rng.randrange(len(windows))]
    return [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in fr[:S]], t[:S].copy()


def extract_val(parts_dir):
    """The validation videos (the train videos listed in clipshots_val_videos.txt, never trained
    on or mined) as .mp4 in DATA/clipshots_val, for choosing thresholds. Skips existing ones.
        python training/clipshots.py --val <dir with ClipShots-a, -b, -c>"""
    names = val_videos()
    dst_dir = VAL
    dst_dir.mkdir(parents=True, exist_ok=True)
    left = {n for n in names if not (dst_dir / n).exists()}
    parts = sorted(Path(parts_dir).glob("ClipShots-?"))
    with tarfile.open(fileobj=io.BufferedReader(Parts(parts), 1 << 20), mode="r|gz") as tar:
        for m in tar:
            bits = m.name.split("/")
            if not left:
                break
            if len(bits) < 5 or bits[2] != "videos" or bits[3] != "train" or bits[4] not in left or not m.isfile():
                continue
            dst = dst_dir / bits[4]
            with open(str(dst) + ".part", "wb") as fh:
                fh.write(tar.extractfile(m).read())
            os.replace(str(dst) + ".part", dst)
            left.discard(bits[4])
    print(f"done: {len(names) - len(left)} of {len(names)} val videos in {dst_dir}; missing {sorted(left)[:5]}")


if __name__ == "__main__":
    if sys.argv[1] == "--val":
        extract_val(sys.argv[2])
    else:
        main(sys.argv[1])
