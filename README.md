<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/leanscene-wordmark-dark.svg">
    <img src="logo/leanscene-wordmark.svg" alt="LeanScene" width="440">
  </picture>
</h1>

A 0.5M-parameter network that finds shot transitions (cuts, dissolves, fades) in video.
It matches TransNet V2 on that model's three benchmarks at 40x fewer operations per frame.

```sh
pip install leanscene
leanscene video.mp4
```

```
video.mp4: 14775 frames, 25.000 fps, 51 shots, 50 transitions
 shot    start      end    start time     end time
    1        0      343  00:00:00.000 00:00:13.760
    2      344     2403  00:00:13.760 00:01:36.160
    ...
```

The weights (2 MB) ship inside the package; nothing is downloaded at run time. The paper is in
preparation.

## Python

```python
import leanscene

result = leanscene.detect("video.mp4")
result.shots         # [(0, 343), (344, 2403), (2414, 3315), ...]  inclusive frame ranges, 0-based
result.transitions   # [Transition(start=344, end=344, kind='cut'), Transition(start=2404, end=2413, kind='gradual'), ...]
result.probs         # (frames, 3) float32: same, cut, fade
result.to_dict()     # JSON-ready, with times in seconds
```

To process many videos, load the model once:

```python
model = leanscene.load_model()            # or load_model(device="cuda")
for path in paths:
    result = leanscene.detect(path, model=model)
```

`detect` also takes an iterable of `HxWx3` uint8 frames in BGR order instead of a path, and
`cut=` / `fade=` thresholds (defaults 0.7 and 0.8, the rule used for every number below).
`frame_probs` returns only the probabilities; `label_frames`, `find_transitions` and
`find_shots` are the steps `detect` applies to them.

A cut is reported on the first frame of the new shot. A gradual transition is the run of
frames inside it, and those frames belong to neither neighbouring shot.

## Command line

| Option | Effect |
|---|---|
| `-f table\|json\|csv` | Output format. `table` (default) and `csv` list shots; `json` has shots and transitions |
| `-o FILE` | Write to a file instead of standard output |
| `--model PATH` | A checkpoint file, to use weights other than the bundled ones |
| `--device DEVICE` | `cpu`, `cuda` or `mps`. Default: `cuda` when available, else `cpu` |
| `--cut X`, `--fade X` | Thresholds (0.7, 0.8). Higher gives fewer false alarms, lower misses less |
| `--probs FILE.npy` | Also save the per-frame probabilities |

Several videos can be given at once. `python -m leanscene` is the same command.

## Results

F1 over transitions with TransNet V2's protocol (two-frame tolerance), one decision rule fixed
before the test sets were scored. TransNet V2 is its published weights run through the same
decoding and scoring code.

| | BBC Planet Earth | ClipShots | RAI | Parameters | MACs per frame |
|---|---|---|---|---|---|
| TransNet V2 (as published) | 96.2 | 77.9 | 93.9 | 7.62 M | 832 M |
| TransNet V2 (same protocol) | 96.1 | 77.7 | 92.6 | | |
| LeanScene, mean of 5 seeds | 97.3 ± 0.2 | 79.1 ± 0.5 | 93.5 ± 1.2 | 0.49 M | 20.9 M |
| **The bundled weights** (one of the 5 seeds) | 97.25 | 79.00 | 93.47 | | |

What these numbers do and do not show:

- A paired bootstrap over videos separates LeanScene from TransNet V2 on **BBC only**
  (+1.13, 95% interval +0.48 to +2.13). On ClipShots and RAI the intervals include zero:
  that is parity, not a win. TransNet V2's published RAI number is above ours.
- The BBC lead shrinks to +0.33 if TransNet V2's threshold is tuned on the test set.
- The BBC lead depends on the hand labels in [`labels/`](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/README.md). Trained without
  them, the model scores 95.4 on BBC, 80.0 on ClipShots and 93.2 on RAI (mean of 3 seeds).
- The test sets were looked at during development. No benchmark here was unseen.

Model-only throughput on an Apple M1 Pro (8 threads, 6,000 frames, decoding excluded):
TransNet V2 159 frames/s on the CPU; LeanScene 570 frames/s in batches of 512 and 1,225 in
batches of 8 on the CPU, and 3,947 on the GPU (MPS). The CPU speed-up is 3.6x to 7.7x, far less
than the 40x reduction in operations. No CUDA timing has been measured.

## Weights

The package ships one set of weights (2 MB, sha256 `091853e5b35fbfd4…`, research run
`seqnet_v24_seed2`). It was trained on synthetic transitions built from single-shot
skateboarding clips, ClipShots training videos, and 1,245 hand labels on skateboarding videos.
The hand labels are in [`labels/`](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/README.md), as YouTube ids and frame indices. The
clips behind the synthetic transitions are not released.

The paper trains the model five times from different random seeds and reports the mean. The
bundled file is the seed with the best F1 on the validation set (906 ClipShots training videos
held out from training), chosen without looking at the test sets.

## How it works

Each frame is resized to 100x100 and encoded once: a four-layer 2D CNN gives a global feature
and a 2x2 grid of features, and two descriptors come straight from the pixels (a 64-bin colour
histogram and an 8x8 colour thumbnail). A temporal head of five residual dilated 1D
convolutions (dilations 1, 2, 4, 8, 16; receptive field 63 frames) sees those features and
their change since the previous frame, and labels every frame *same*, *cut* or *fade*.

The video is streamed, so memory stays flat on long files: frames are encoded in batches and
the head runs over 4,096 frames at a time with 64 frames of context on each side, which gives
the same output as a single pass.

## Labels

[`labels/`](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/README.md) holds the 1,245 hand labels (cuts, fades and confirmed
non-transitions on 466 skateboarding videos) the model was trained on.

## Browser demo

[`demo/`](https://github.com/lucasjacks0n/LeanScene/blob/main/demo/README.md) runs the same model in a browser tab with ONNX Runtime Web: drop a
video on the page and it shows the transitions on a timeline. Nothing is uploaded.

## Reproducing the numbers

[`eval/`](https://github.com/lucasjacks0n/LeanScene/blob/main/eval/README.md) scores the bundled weights on the three benchmarks with the
paper's protocol.

## Limits

- Videos are decoded with OpenCV. Times are frame index divided by the container's frame
  rate, so they drift on variable-frame-rate files; frame indices are exact.
- `--device mps` is not the default on Apple hardware: a process that fed MPS many different
  input lengths kept growing in memory in our runs.
- Training code is not in this repository.

## Development

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,export]"
.venv/bin/python -m pytest
```

`.venv/` is about 1 GB (mostly PyTorch), is gitignored, and is safe to delete; the two commands
above recreate it. The `export` extra is only needed for `demo/export_onnx.py`.

## Citation

```bibtex
@misc{jackson2026leanscene,
  title  = {LeanScene: A 0.5M-Parameter Network for Fast Shot Transition Detection},
  author = {Lucas Jackson},
  year   = {2026},
  note   = {In preparation}
}
```

## Licence

- Code and model weights: MIT, see [LICENSE](https://github.com/lucasjacks0n/LeanScene/blob/main/LICENSE). `eval/evaluate.py` contains two functions
  adapted from [TransNet V2](https://github.com/soCzech/TransNetV2) (MIT).
- The labels in [`labels/`](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/README.md): [CC BY 4.0](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/LICENSE). Use them freely, with credit.
