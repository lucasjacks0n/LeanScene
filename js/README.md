# leanscene

Fast shot transition detection for the browser and Node.js: a 0.39M-parameter network that
finds cuts, dissolves and fades in video. It scores higher than TransNet V2 on all three of
that model's benchmarks with 115x fewer operations per frame.

```sh
npm install leanscene
```

**In the browser**, give it a file. The video is decoded and analyzed in the page, on the GPU
when there is one; nothing is uploaded.

```ts
import { detect } from "leanscene";

input.onchange = async () => {
  const { shots, transitions } = await detect(input.files[0]);
};
```

**In Node**, give it a path (ffmpeg must be installed), or use the command line:

```ts
import { detect } from "leanscene";

const { shots, transitions } = await detect("video.mp4");
```

```sh
npx leanscene video.mp4
```

```
video.mp4: 14775 frames, 25.000 fps, 51 shots, 50 transitions
 shot    start      end    start time     end time
    1        0      343  00:00:00.000 00:00:13.760
    2      344     2403  00:00:13.760 00:01:36.160
    ...
```

The same `import` works in both: bundlers pick the browser build, Node picks the Node build.
The weights (1.6 MB) ship inside the package. TypeScript types are included. ES modules; Node
18 or later.

The model, the paper and the Python package are at
[github.com/lucasjacks0n/LeanScene](https://github.com/lucasjacks0n/LeanScene), and there is a
[live demo](https://lucasjackson.dev/leanscene/).

## The result

```ts
const result = await detect(source);
result.shots;        // [[0, 343], [344, 2403], [2414, 3315], ...]  inclusive frame ranges, 0-based
result.transitions;  // [{ start: 344, end: 344, kind: "cut" }, { start: 2404, end: 2413, kind: "gradual" }, ...]
result.probs;        // Float32Array, three values per frame: same, cut, fade
result.fps;          // 25, or null when unknown
JSON.stringify(result);  // shots and transitions with times in seconds; probs left out
```

A cut is reported on the first frame of the new shot. A gradual transition is the run of
frames inside it, and those frames belong to neither neighboring shot.

To process many videos, load the model once and pass it along:

```ts
import { detect, loadModel } from "leanscene";

const model = await loadModel();
for (const video of videos) {
  const result = await detect(video, { model });
}
```

## In the browser

There is nothing to configure in a bundler: Vite (dev server and production build) and esbuild
were tried, with no settings. `detect` takes:

| Source | |
|---|---|
| `File` or `Blob` | From `<input type="file">`, a drop, or a `fetch` |
| A URL (`string` or `URL`) | Fetched; another origin must allow it (CORS) |
| `ArrayBuffer` or `Uint8Array` | The bytes of a file |
| A `<video>` element | Its file is fetched again by URL, so the same rule applies |
| Your own frames | See below |

Show progress and let the user cancel:

```ts
const stop = new AbortController();
const result = await detect(file, {
  progress: (frames, total) => bar.value = total ? frames / total : 0,
  signal: stop.signal,
});
```

What it needs and what it loads:

- **WebCodecs**, to decode: current Chrome, Edge, Safari and Firefox have it. Only Chrome was
  tried. Files must be **MP4 or MOV** with a codec the browser can decode (H.264, and usually
  HEVC, VP9 or AV1).
- **WebGPU** when the browser has it, otherwise WebAssembly on the CPU. `loadModel({ backend:
  "wasm" })` forces the CPU.
- ONNX Runtime's `.wasm` file (15 MB for the CPU, 30 MB with WebGPU) is fetched once from
  **cdn.jsdelivr.net** on first use and cached by the browser. To serve it yourself, copy
  `node_modules/onnxruntime-web/dist/*.wasm` into your site and pass
  `loadModel({ wasmPaths: "/your/path/" })`.
- The weights add a 2.1 MB script (1.6 MB gzipped) that is loaded when the model is.

Detection runs on the page's thread between frames. For a page that must stay smooth during a
long video, call it from a Web Worker: everything it uses exists in workers, though that was
not tried.

| `loadModel` option (browser) | Default | Meaning |
|---|---|---|
| `backend` | WebGPU if present | `"webgpu"` or `"wasm"` |
| `wasmPaths` | jsDelivr | URL prefix for ONNX Runtime's `.wasm`, or `false` to leave ONNX Runtime's own setting |
| `threads` | 1, or up to 4 on a cross-origin isolated page | WebAssembly threads |
| `weights` | bundled | `{ features, head, meta }`: the two `.onnx` files as URLs or bytes, and `meta.json`'s content |

### Your own frames

`detect` and `frameProbs` also take frames, as any iterable or async iterable. A frame is a
`Uint8Array` of 30,000 bytes: 100x100 pixels, three bytes each in **B, G, R** order, row by
row. In the browser `frameFrom` makes one from anything a canvas can draw: a `<video>` at its
current time, an image, a canvas, an `ImageBitmap`, a `VideoFrame`.

```ts
import { detect, frameFrom } from "leanscene";

const frames = [];
// for each frame you have:
frames.push(frameFrom(canvasOrVideoOrBitmap));
const result = await detect(frames);
```

## In Node

Videos are decoded by **ffmpeg, which must be installed**: on the `PATH`, or as the
[`ffmpeg-static`](https://www.npmjs.com/package/ffmpeg-static) package next to this one, or
anywhere with the `ffmpeg` option. Any format ffmpeg reads works. A pipeline that already
decodes video can pass frames instead (the format above) and needs no ffmpeg.

```ts
async function* frames() { /* yield one 100x100 BGR Uint8Array per frame */ }
const result = await detect(frames());
```

| `loadModel` option (Node) | Default | Meaning |
|---|---|---|
| `weights` | bundled | A directory with `features.onnx`, `head.onnx` and `meta.json` |
| `threads` | ONNX Runtime decides | Threads for one inference call |
| `executionProviders` | the CPU | Passed to ONNX Runtime (`["cuda"]`, `["coreml"]`, ...); only the CPU has been tried |

### Command line

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

## API

| Function | Returns |
|---|---|
| `detect(source, options?)` | `{ frames, fps, transitions, shots, probs }` |
| `frameProbs(source, options?)` | The probabilities only: `Float32Array`, three per frame |
| `labelFrames(probs, { cut, fade, close }?)` | One label per frame: 0 same, 1 cut, 2 fade |
| `findTransitions(labels)`, `findShots(transitions, frames)` | The two steps `detect` applies after `labelFrames` |
| `loadModel(options?)` | A model to pass as `options.model` |
| `decodeFrames(file)` | An async iterator of frames, as `detect` decodes them. Browser: a `Blob`. Node: a path |
| `frameFrom(image)` | Browser only: one frame from anything a canvas can draw |
| `frameRate(path)`, `ffmpegPath()` | Node only: frames per second from `ffprobe` (or `null`); the ffmpeg in use |

| `detect` option | Default | Meaning |
|---|---|---|
| `model` | loaded per call | A model from `loadModel()` |
| `cut`, `fade` | 0.7, 0.8 | Thresholds. Higher gives fewer false alarms, lower misses less. The defaults are the rule used for every number in the paper |
| `close` | 8 | Fade frames at most this many frames apart are joined |
| `batch` | 8 on a CPU, 64 on WebGPU | Frames encoded per call |
| `window` | 4096 | Frames the temporal head labels per call |
| `progress` | none | Called with the frames done so far (and, in the browser, the file's frame count when it says) |
| `signal` | none | An `AbortSignal` that stops the decoder |
| `ffmpeg` | see above | Node only: the ffmpeg program |

`leanscene/browser` and `leanscene/node` name one build explicitly, for a setup that picks
the wrong one or for TypeScript, whose types for `"leanscene"` are the Node ones unless
`customConditions: ["browser"]` is set.

## Same results everywhere?

Close, not identical. The network is the same in all three: on identical frames the Python
and npm packages' probabilities differ by at most 0.000003. The decoders differ: Python
resizes frames with OpenCV, Node with ffmpeg and the browser with a canvas, which do not
produce the same pixels. Frame counts and frame indices are the same.

**Node against Python**, measured on 2026-11-18 with `scripts/parity.py` on 8 skateboarding
videos (71,507 frames; H.264, AV1 and VP9; 360p to 720p), Apple M1 Pro, ffmpeg 7.1.1, CPU:

| | Python | Node |
|---|---|---|
| Frames decoded | 71,507 | 71,507 (the same count on every video) |
| Transitions found | 524 | 527 |
| Found by one only (two-frame tolerance) | 3 | 6 |
| Frames per second, end to end | 1,106 | 1,922 |

**Browser against Node**, measured on 2026-11-22 with `scripts/browser_parity.mjs` on 9 videos
(the 8 above and the demo's sample; 72,707 frames), headless Chrome on the same machine:

| | Node | Browser, WebGPU | Browser, WebAssembly (1 thread) |
|---|---|---|---|
| Frames decoded | 72,707 | 72,707 (the same count on every video) | 72,707 |
| Transitions found | 546 | 545 | 545 |
| Found by one only (two-frame tolerance) | 4 | 3 | 3 |
| Frames per second, end to end | about 1,800 | 707 | 524 |

Per-frame probabilities differed by 0.002 on average in both comparisons and 0.35 at most. The
transitions that differ are ones near a threshold.

**Not measured:** either npm build on the paper's benchmarks (BBC, ClipShots, RAI). The
paper's F1 numbers were measured with the Python pipeline; expect these within the
differences above, but they have not been scored. Speeds are one run per video on one machine.
Not tried: Safari, Firefox, phones, video above 720p in the browser, and importing the package
from a CDN without a bundler.

## How it works

1. Every frame of the first video track is decoded and scaled to 100x100, without dropping or
   duplicating frames, so frame indices match a frame-by-frame reader such as OpenCV. Node
   runs `ffmpeg`; the browser demuxes with `mp4box` and decodes with WebCodecs.
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
- Only the first video track is read. Rotation metadata is ignored in the browser (frames are
  squashed to 100x100 either way).
- Browser: MP4 and MOV only, and a file whose index is at its end is read to that point before
  the first frame comes out.
- Node: `fps` is `null` when `ffprobe` is not beside `ffmpeg` (the `ffmpeg-static` package has
  no `ffprobe`); shots and transitions are unaffected.
- Installing brings both ONNX Runtimes: `onnxruntime-node` (about 290 MB on disk) and
  `onnxruntime-web` (about 145 MB). A bundle for the browser contains neither; it loads the
  parts it uses at run time.

## Development

From the repository root:

```sh
cd js
npm install          # the ONNX Runtimes, mp4box, TypeScript, and esbuild + puppeteer-core for the browser tests
npm test             # compiles to dist/, then runs test/
```

The browser tests bundle the browser build and run it in headless Chrome on WebAssembly; they
are skipped when Chrome (`CHROME_PATH`, or the usual place) or ffmpeg is missing.

| Path | What it is | Size | Safe to delete? |
|---|---|---|---|
| `js/node_modules/` | Dependencies, written by `npm install`; read by everything here | 515 MB on 2026-11-22: `onnxruntime-node` 290 MB and `onnxruntime-web` 145 MB (each ships binaries for every platform) | Yes; `npm install` restores it (network needed) |
| `js/dist/` | The compiled package, written by `npm run build`; read by the tests, the scripts and `npm publish`. `dist/browser/weights-data.js` (2.1 MB) is the two graphs of `weights/` as base64, written by `scripts/embed_weights.mjs` | 2.2 MB | Yes; `npm run build` restores it |

Both are gitignored. `weights/` is tracked: `scripts/export_onnx.py [checkpoint.pt]`
regenerates it from the Python package's weights (it needs `pip install -e ".[export]"`) and
checks the graphs against PyTorch. `scripts/parity.py VIDEO ...` and
`scripts/browser_parity.mjs webgpu|wasm VIDEO ...` produce the comparisons above; both write
only to a temporary directory that they remove.

## License

MIT. The model weights are covered by the same license.
