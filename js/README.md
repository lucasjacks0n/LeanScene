# leanscene

Fast shot transition detection for Node.js: a 0.39M-parameter network that finds cuts,
dissolves and fades in video. It scores higher than TransNet V2 on all three of that model's
benchmarks with 115x fewer operations per frame, and this package runs it at about 1,900
frames per second on a laptop CPU, decoding included.

```sh
npm install leanscene
npx leanscene video.mp4
```

```
video.mp4: 14775 frames, 25.000 fps, 51 shots, 50 transitions
 shot    start      end    start time     end time
    1        0      343  00:00:00.000 00:00:13.760
    2      344     2403  00:00:13.760 00:01:36.160
    ...
```

The weights (1.6 MB) ship inside the package; nothing is downloaded at run time. Videos are
decoded by **ffmpeg, which must be installed**: on the `PATH`, or as the
[`ffmpeg-static`](https://www.npmjs.com/package/ffmpeg-static) package next to this one, or
anywhere with the `ffmpeg` option. TypeScript types are included. Node 18 or later, ES modules.

The model, the paper and the Python package are at
[github.com/lucasjacks0n/LeanScene](https://github.com/lucasjacks0n/LeanScene).

## Use

```ts
import { detect, loadModel } from "leanscene";

const result = await detect("video.mp4");
result.shots;        // [[0, 343], [344, 2403], [2414, 3315], ...]  inclusive frame ranges, 0-based
result.transitions;  // [{ start: 344, end: 344, kind: "cut" }, { start: 2404, end: 2413, kind: "gradual" }, ...]
result.probs;        // Float32Array, three values per frame: same, cut, fade
result.fps;          // 25, or null when unknown
JSON.stringify(result);  // shots and transitions with times in seconds; probs left out
```

To process many videos, load the model once:

```ts
const model = await loadModel();
for (const path of paths) {
  const result = await detect(path, { model });
}
```

A cut is reported on the first frame of the new shot. A gradual transition is the run of
frames inside it, and those frames belong to neither neighboring shot.

### Your own frames

`detect` and `frameProbs` also take frames instead of a path, as any iterable or async
iterable, so a pipeline that already decodes video does not need ffmpeg here. A frame is a
`Uint8Array` of 30,000 bytes: 100x100 pixels, three bytes each in **B, G, R** order, row by row.

```ts
import { detect } from "leanscene";

async function* frames() { /* yield one 100x100 BGR Uint8Array per frame */ }
const result = await detect(frames());
```

### API

| Function | Returns |
|---|---|
| `detect(source, options?)` | `{ frames, fps, transitions, shots, probs }` |
| `frameProbs(source, options?)` | The probabilities only: `Float32Array`, three per frame |
| `labelFrames(probs, { cut, fade, close }?)` | One label per frame: 0 same, 1 cut, 2 fade |
| `findTransitions(labels)`, `findShots(transitions, frames)` | The two steps `detect` applies after `labelFrames` |
| `loadModel({ weights, threads, executionProviders }?)` | A model to pass as `options.model` |
| `decodeFrames(path, { ffmpeg, signal }?)` | An async iterator of frames, as `detect` decodes them |
| `frameRate(path)` | Frames per second from `ffprobe`, or `null` |

| `detect` option | Default | Meaning |
|---|---|---|
| `model` | loaded per call | A model from `loadModel()` |
| `cut`, `fade` | 0.7, 0.8 | Thresholds. Higher gives fewer false alarms, lower misses less. The defaults are the rule used for every number in the paper |
| `close` | 8 | Fade frames at most this many frames apart are joined |
| `batch`, `window` | 8, 4096 | Frames encoded per call, and frames the temporal head labels per call |
| `progress` | none | Called with the number of frames read so far |
| `ffmpeg` | see above | The ffmpeg program |
| `signal` | none | An `AbortSignal` that stops the decoder |

`executionProviders` is passed to ONNX Runtime (`["cuda"]`, `["coreml"]`, ...); only the CPU
has been tried.

## Command line

`npx leanscene VIDEO [VIDEO ...]`, or `leanscene` after a global install.

| Option | Effect |
|---|---|
| `-f table\|json\|csv` | Output format. `table` (default) and `csv` list shots; `json` has shots and transitions |
| `-o FILE` | Write to a file instead of standard output |
| `--cut X`, `--fade X` | Thresholds (0.7, 0.8) |
| `--probs FILE.npy` | Also save the per-frame probabilities, as a NumPy file |
| `--weights DIR` | Other weights: a directory with `features.onnx`, `head.onnx` and `meta.json` |
| `--threads N` | Threads for the model |
| `--ffmpeg PATH` | The ffmpeg program |
| `-q`, `--quiet` | No progress line (it goes to standard error, and only in a terminal) |

The Python package installs a command of the same name; the table, JSON and CSV layouts are the same.

## Same results as the Python package?

Close, not identical. The network is the same: on identical frames the two packages'
probabilities differ by at most 0.000003. The decoders differ: Python resizes frames with
OpenCV and this package with ffmpeg, which do not produce the same pixels.

Measured on 2026-11-18 with `scripts/parity.py` on 8 skateboarding videos (71,507 frames;
H.264, AV1 and VP9; 360p to 720p), Apple M1 Pro, ffmpeg 7.1.1, onnxruntime-node on the CPU:

| | Python | npm |
|---|---|---|
| Frames decoded | 71,507 | 71,507 (the same count on every video) |
| Transitions found | 524 | 527 |
| Found by one package only (two-frame tolerance) | 3 | 6 |
| Frames per second, end to end | 1,106 | 1,922 |

Per-frame probabilities differed by 0.0018 on average and 0.33 at most. The transitions that
differ are ones near a threshold.

**Not measured:** this package on the paper's benchmarks (BBC, ClipShots, RAI). The paper's F1
numbers were measured with the Python pipeline; expect this one within the difference above,
but it has not been scored. Speed is one run per video on one machine, model loading included.

## How it works

1. `ffmpeg` decodes the first video stream and scales every frame to 100x100, without dropping
   or duplicating frames, so frame indices match a frame-by-frame reader such as OpenCV.
2. Each frame is encoded once: a four-layer 2D CNN (`features.onnx`) gives a global feature, a
   2x2 grid of features and an 8x8 color thumbnail; a 64-bin color histogram is computed here.
3. A temporal head of five residual dilated 1D convolutions (`head.onnx`, receptive field 63
   frames) labels every frame *same*, *cut* or *fade*. It runs over 4,096 frames at a time with
   64 frames of context on each side, which gives the same output as a single pass, so memory
   stays flat on long videos.
4. The rule: cut at probability 0.7 or more, fade at 0.8 or more, fade runs joined over gaps of
   up to 8 frames.

Nothing is written to disk and there is no state to resume.

## Limits

- Times are frame index divided by the average frame rate, so they drift on
  variable-frame-rate files; frame indices are exact.
- `fps` is `null` when `ffprobe` is not beside `ffmpeg` (the `ffmpeg-static` package has no
  `ffprobe`); shots and transitions are unaffected.
- Only the first video stream is read.
- The browser is not supported by this package; the repository's `demo/` runs the same model
  in a browser tab.

## Development

From the repository root:

```sh
cd js
npm install          # onnxruntime-node and TypeScript
npm test             # compiles to dist/, then runs test/
```

| Path | What it is | Size | Safe to delete? |
|---|---|---|---|
| `js/node_modules/` | Dependencies, written by `npm install`; read by everything here | 321 MB on 2026-11-18, nearly all `onnxruntime-node` (it ships binaries for every platform) | Yes; `npm install` restores it (network needed) |
| `js/dist/` | The compiled package, written by `npm run build`; read by the tests, the scripts and `npm publish` | 52 KB | Yes; `npm run build` restores it |

Both are gitignored. `weights/` is tracked: `scripts/export_onnx.py [checkpoint.pt]`
regenerates it from the Python package's weights (it needs `pip install -e ".[export]"`) and
checks the graphs against PyTorch. `scripts/parity.py VIDEO ...` produces the comparison above.

## License

MIT. The model weights are covered by the same license.
