# Evaluation

`evaluate.py` scores LeanScene on the three benchmarks with the paper's protocol: TransNet V2's
metric (F1 over transitions, two-frame tolerance, counts summed over the videos of a set) and
the fixed decision rule (cut >= 0.7, fade >= 0.8).

Run from the repository root, with the package installed:

```sh
python eval/evaluate.py rai /data/rai
python eval/evaluate.py bbc /data/bbc
python eval/evaluate.py clipshots /data/clipshots/videos/test /data/clipshots/annotations/test.json
```

| Option | Effect |
|---|---|
| `--model PATH` | A checkpoint file, to score weights other than the bundled ones |
| `--device DEVICE` | `cpu`, `cuda` or `mps` |
| `--cut X`, `--fade X` | Thresholds, default 0.7 and 0.8 |
| `-v` | Print every video |

Nothing is written to disk and there is no state to resume; a run starts from the first video.

## Dataset layout

The datasets are not distributed here. Each is expected as its authors ship it:

| Set | Layout | Ground truth |
|---|---|---|
| `bbc` | `bbc_<NN>.mp4` anywhere under the root, and `annotations/shots/<NN>_*.txt` | One shot per line. Read with TransNet V2's convention (frame numbers + 1) |
| `rai` | `<id>.mp4` and `<id>_gt.txt` side by side | One shot per line, first and last frame, 0-based |
| `clipshots` | A folder of test videos, and `annotations/test.json` | The `transitions` and `frame_num` fields |

## Expected numbers

For the bundled weights:

| Set | Videos | Transitions | F1 | Precision | Recall |
|---|---|---|---|---|---|
| BBC Planet Earth | 11 | 4,844 | 97.45 | 97.88 | 97.03 |
| ClipShots test | 500 | 7,209 | 78.82 | 73.93 | 84.39 |
| RAI | 10 | 985 | 93.03 | 93.99 | 92.08 |

These come from the research code (PyTorch 2.8, OpenCV 4.8, on Apple MPS).

**What has been checked with this script:** RAI, on 2026-11-17 on the CPU, gave F1 93.03,
precision 93.99, recall 92.08 (true positives 907, false alarms 58, misses 78), the same as
above. BBC and ClipShots have **not** been re-run with this script.

A different OpenCV or FFmpeg build can decode a few frames differently, which can move a
result in the second decimal.
