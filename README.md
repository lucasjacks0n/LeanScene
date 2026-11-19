<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/leanscene-wordmark-dark.svg">
    <img src="logo/leanscene-wordmark.svg" alt="LeanScene" width="440">
  </picture>
</h1>

A 0.39M-parameter network that finds shot transitions (cuts, dissolves, fades) in video.
It scores higher than TransNet V2 on all three of that model's benchmarks, with 115x fewer
operations per frame.

**[Try it in your browser](https://lucasjackson.dev/leanscene/)**: drop a video and see every
transition on a timeline. It runs in the tab; nothing is uploaded.

```sh
# Python
pip install git+https://github.com/lucasjacks0n/LeanScene
leanscene video.mp4

# Node.js
npm install leanscene
npx leanscene video.mp4
```

The Node.js package is [on npm](https://www.npmjs.com/package/leanscene). A PyPI release
(`pip install leanscene`) is on its way; until then, install the Python package from GitHub as above.

```
video.mp4: 14775 frames, 25.000 fps, 51 shots, 50 transitions
 shot    start      end    start time     end time
    1        0      343  00:00:00.000 00:00:13.760
    2      344     2403  00:00:13.760 00:01:36.160
    ...
```

The weights (1.6 MB) ship inside the package; nothing is downloaded at run time. The paper is in
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
| `-q`, `--quiet` | No progress line |

In a terminal it shows a live progress line (frames done, frames per second, time left). The line goes to
standard error, so piped or redirected output stays clean. From Python, pass `progress=` a function that
takes the number of frames read so far.

Several videos can be given at once. `python -m leanscene` is the same command.

## Node.js

[`js/`](https://github.com/lucasjacks0n/LeanScene/blob/main/js/README.md) is the same model as an [npm package](https://www.npmjs.com/package/leanscene) for Node, with TypeScript types and
the same command line. It decodes with ffmpeg and runs the network with ONNX Runtime.

```sh
npm install leanscene
npx leanscene video.mp4
```

```ts
import { detect } from "leanscene";
const { shots, transitions } = await detect("video.mp4");
```

On 8 test videos it found 527 transitions where the Python package found 524, at 1,922 frames
per second end to end on a laptop CPU against 1,106; its README has the comparison.

## Results

LeanScene scores higher than TransNet V2 on all three benchmarks, at a twentieth of the size and with 115x
fewer operations per frame.

| | BBC Planet Earth | ClipShots | RAI | Parameters | MACs per frame |
|---|---|---|---|---|---|
| TransNet V2 | 96.1 | 77.7 | 92.6 | 7.62 M | 832 M |
| **LeanScene** (mean of 5 seeds) | **97.4** ± 0.1 | **79.1** ± 0.2 | **93.7** ± 0.5 | **0.39 M** | **7.2 M** |
| The bundled weights (one of the 5 seeds) | 97.45 | 78.82 | 93.03 | | |
| LeanScene at full width (not shipped; mean of 5 seeds) | 97.3 ± 0.2 | 79.1 ± 0.5 | 93.5 ± 1.2 | 0.49 M | 20.9 M |

F1 over transitions with TransNet V2's own protocol (two-frame tolerance). Both models are run
through the same decoding and scoring code, and LeanScene's decision rule was fixed before the
test sets were scored.

It is fast as well as small. On an Apple M1 Pro the full-width model alone runs at 1,225
frames/s on the CPU and 3,947 on the GPU, against 159 for TransNet V2 on the CPU. The shipped
model does about a third of the full-width model's operations.

<details>
<summary>Fine print</summary>

- TransNet V2's paper reports 96.2, 77.9 and 93.9. The row above is its published weights
  measured by us; we could not reproduce its RAI number.
- Every one of the five seeds is above TransNet V2 on every benchmark.
- A paired bootstrap over videos puts the BBC lead clear of zero (+1.28, 95% interval +0.50 to
  +2.47). The ClipShots and RAI leads (+1.36 and +1.13) have intervals that include zero.
- The BBC lead shrinks to +0.48 if TransNet V2's threshold is tuned on the test set, and it
  depends on the hand labels in [`labels/`](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/README.md): trained without them, the
  full-width model scores 95.4 on BBC, 80.0 on ClipShots and 93.2 on RAI (mean of 3 seeds).
- The test sets were looked at during development.
- Speeds are model only (8 threads, 6,000 frames, decoding excluded); LeanScene's CPU figure is
  with batches of 8 frames (570 frames/s with batches of 512). They were measured on the
  full-width model; the shipped model has not been timed yet, and no CUDA timing has been measured.

</details>

## Weights

The package ships one set of weights (1.6 MB, sha256 `b0d1cc2a77f8aa91…`, research run
`seqnet_w05_seed2`). It was trained on synthetic transitions built from single-shot
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

Live at **[lucasjackson.dev/leanscene](https://lucasjackson.dev/leanscene/)**.
[`demo/`](https://github.com/lucasjacks0n/LeanScene/blob/main/demo/README.md) runs the same model in a browser tab with ONNX Runtime Web: drop a
video on the page and it shows the transitions on a timeline. Nothing is uploaded.

## Reproducing the numbers

[`eval/`](https://github.com/lucasjacks0n/LeanScene/blob/main/eval/README.md) scores the bundled weights on the three benchmarks with the
paper's protocol.

## Training

[`training/`](https://github.com/lucasjacks0n/LeanScene/blob/main/training/README.md) is the code that trained the bundled weights: the trainer, the
synthetic transitions, and the mining and cleaning of hard negatives, with the command for
each step of the paper's recipe. The single-shot clips behind the synthetic sequences are not
distributed, so bring your own to train.

## Limits

- Videos are decoded with OpenCV. Times are frame index divided by the container's frame
  rate, so they drift on variable-frame-rate files; frame indices are exact.
- `--device mps` is not the default on Apple hardware: a process that fed MPS many different
  input lengths kept growing in memory in our runs.

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
  title  = {LeanScene: A 0.39M-Parameter Network for Fast Shot Transition Detection},
  author = {Lucas Jackson},
  year   = {2026},
  note   = {In preparation}
}
```

## Licence

- Code and model weights: MIT, see [LICENSE](https://github.com/lucasjacks0n/LeanScene/blob/main/LICENSE). `eval/evaluate.py` contains two functions
  adapted from [TransNet V2](https://github.com/soCzech/TransNetV2) (MIT).
- The labels in [`labels/`](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/README.md): [CC BY 4.0](https://github.com/lucasjacks0n/LeanScene/blob/main/labels/LICENSE). Use them freely, with credit.
