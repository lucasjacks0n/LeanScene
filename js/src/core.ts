/**
 * The parts of LeanScene that do not depend on where it runs: turning frames into the network's
 * input, the streaming pass over a video, and the result. Node and the browser each supply a
 * Runtime (the two ONNX graphs on their ONNX Runtime) and a decoder.
 */

import { findShots, findTransitions, labelFrames, type RuleOptions, type Shot, type Transition } from "./rule.js";

/** Frames are FRAME_SIZE x FRAME_SIZE before the network sees them. */
export const FRAME_SIZE = 100;
/** Bytes in one frame: FRAME_SIZE x FRAME_SIZE pixels, three bytes each, in B, G, R order. */
export const FRAME_BYTES = FRAME_SIZE * FRAME_SIZE * 3;

const PIXELS = FRAME_SIZE * FRAME_SIZE;

/** weights/meta.json */
export interface Meta {
  checkpoint: string;
  dim: number;
  feat: number;
  size: number;
  histBins: number;
  receptiveField: number;
}

/** The network on some ONNX Runtime. loadModel() returns one. */
export interface Runtime {
  /** Values per frame the temporal head reads. */
  readonly dim: number;
  /** Number of frames that can influence one frame's label. */
  readonly receptiveField: number;
  /** Name of the checkpoint the graphs were exported from. */
  readonly checkpoint: string;
  /** @internal */
  readonly meta: Meta;
  /** @internal x: n x 3 x FRAME_SIZE x FRAME_SIZE in [0,1], B, G, R planes. Returns n x (dim - histogram bins). */
  runFeatures(x: Float32Array, n: number): Promise<Float32Array>;
  /** @internal features: frames x dim. Returns frames x 3 probabilities. */
  runHead(features: Float32Array, frames: number): Promise<Float32Array>;
}

export function checkMeta(meta: Meta): Meta {
  if (meta.size !== FRAME_SIZE) throw new Error(`these weights expect ${meta.size}x${meta.size} frames, not ${FRAME_SIZE}`);
  return meta;
}

// Byte -> float, and byte -> histogram bin, computed in float32 the way the PyTorch model does.
const UNIT = new Float32Array(256);
const BIN = new Uint8Array(256);
function tables(bins: number): void {
  const scale = Math.fround(bins - 0.001);
  for (let v = 0; v < 256; v++) {
    UNIT[v] = v / 255;
    BIN[v] = Math.trunc(Math.fround(UNIT[v] * scale));
  }
}

/**
 * Per-frame features of a batch. `frames` are FRAME_BYTES-long, B, G, R per pixel, row by row.
 * Returns frames.length x model.dim values.
 */
export async function frameFeatures(model: Runtime, frames: Uint8Array[]): Promise<Float32Array> {
  const n = frames.length;
  const { dim, feat, histBins: bins } = model.meta;
  if (BIN[255] !== bins - 1) tables(bins);
  const nb = bins ** 3;
  const x = new Float32Array(n * 3 * PIXELS);
  const out = new Float32Array(n * dim);
  for (let k = 0; k < n; k++) {
    const f = frames[k];
    if (f.length !== FRAME_BYTES) throw new Error(`a frame must be ${FRAME_BYTES} bytes (${FRAME_SIZE}x${FRAME_SIZE} BGR), got ${f.length}`);
    const b = k * 3 * PIXELS, g = b + PIXELS, r = g + PIXELS, h = k * dim + feat;
    for (let i = 0, j = 0; i < PIXELS; i++, j += 3) {
      x[b + i] = UNIT[f[j]];
      x[g + i] = UNIT[f[j + 1]];
      x[r + i] = UNIT[f[j + 2]];
      out[h + (BIN[f[j]] * bins + BIN[f[j + 1]]) * bins + BIN[f[j + 2]]]++;
    }
    for (let i = 0; i < nb; i++) out[h + i] /= PIXELS;
  }
  const y = await model.runFeatures(x, n);
  const rest = dim - feat - nb;  // the graph returns [CNN mean | thumbnail | grid]; the histogram goes after the mean
  for (let k = 0; k < n; k++) {
    const src = k * (dim - nb), dst = k * dim;
    out.set(y.subarray(src, src + feat), dst);
    out.set(y.subarray(src + feat, src + feat + rest), dst + feat + nb);
  }
  return out;
}

/** Probabilities of same / cut / fade for a run of frames, from their features: three values per frame. */
export function headProbs(model: Runtime, features: Float32Array, frames: number): Promise<Float32Array> {
  return model.runHead(features, frames);
}

export interface StreamOptions {
  /** Frames encoded per call (default 8 in Node, 64 in the browser). */
  batch?: number;
  /** Frames the temporal head labels per call (default 4096). */
  window?: number;
  /** Called with the number of frames read so far, after each batch. */
  progress?: (frames: number) => void;
}

/**
 * Per-frame probabilities of same / cut / fade for a stream of frames: three values per frame.
 *
 * Frames are encoded in batches and the temporal head runs over `window` frames at a time with
 * more context on each side than its receptive field, so memory stays flat on long videos and
 * the result equals a single pass up to float rounding.
 */
export async function streamProbs(model: Runtime, frames: AsyncIterable<Uint8Array> | Iterable<Uint8Array>,
                                  options: StreamOptions = {}): Promise<Float32Array> {
  const { batch = 8, window = 4096, progress } = options;
  const D = model.dim;
  const ctx = Math.max(64, (model.receptiveField >> 1) + 1);
  const out: Float32Array[] = [];
  let feats = new Float32Array((window + 2 * ctx + batch) * D);  // features not yet labeled; the first `done` rows are left context
  let held = 0, done = 0, seen = 0;

  const encode = async (pending: Uint8Array[]) => {
    const f = await frameFeatures(model, pending);
    if ((held + pending.length) * D > feats.length) {
      const grown = new Float32Array(Math.max(feats.length * 2, (held + pending.length) * D));
      grown.set(feats.subarray(0, held * D));
      feats = grown;
    }
    feats.set(f, held * D);
    held += pending.length;
    seen += pending.length;
    progress?.(seen);
  };
  const label = async (final: boolean) => {
    const hi = final ? held : held - ctx;
    if (hi <= done) return;
    const p = await headProbs(model, feats.subarray(0, held * D), held);
    out.push(p.slice(done * 3, hi * 3));
    const keep = Math.max(0, hi - ctx);
    feats.copyWithin(0, keep * D, held * D);
    held -= keep;
    done = hi - keep;
  };

  let pending: Uint8Array[] = [];
  for await (const frame of frames as AsyncIterable<Uint8Array>) {
    pending.push(frame);
    if (pending.length === batch) {
      await encode(pending);
      pending = [];
      if (held - done >= window + ctx) await label(false);
    }
  }
  if (pending.length) await encode(pending);
  await label(true);
  const total = out.reduce((a, p) => a + p.length, 0);
  const probs = new Float32Array(total);
  let at = 0;
  for (const p of out) {
    probs.set(p, at);
    at += p.length;
  }
  return probs;
}

export interface Result {
  frames: number;
  /** Frames per second; null when the source is not a file or the file does not say. */
  fps: number | null;
  transitions: Transition[];
  shots: Shot[];
  /** Three values per frame: same, cut, fade. */
  probs: Float32Array;
  /**
   * JSON-ready (JSON.stringify uses it). Times are in seconds and present only when fps is
   * known; a shot's end time is the end of its last frame. `probs` is left out.
   */
  toJSON(): ResultJSON;
}

export interface Span {
  start: number;
  end: number;
  start_time?: number;
  end_time?: number;
}

export interface ResultJSON {
  frames: number;
  fps: number | null;
  shots: Span[];
  transitions: (Span & { kind: "cut" | "gradual" })[];
}

function toJSON(this: Result): ResultJSON {
  const { fps } = this;
  const span = <T extends Span>(d: T): T => {
    if (fps) {
      d.start_time = Math.round((d.start / fps) * 1e4) / 1e4;
      d.end_time = Math.round(((d.end + 1) / fps) * 1e4) / 1e4;
    }
    return d;
  };
  return {
    frames: this.frames,
    fps,
    shots: this.shots.map(([start, end]) => span({ start, end })),
    transitions: this.transitions.map(t => span({ ...t })),
  };
}

/** The decision rule applied to per-frame probabilities, as detect() returns it. */
export function resultFrom(probs: Float32Array, fps: number | null, rule: RuleOptions = {}): Result {
  const frames = probs.length / 3;
  const transitions = findTransitions(labelFrames(probs, rule));
  return { frames, fps, transitions, shots: findShots(transitions, frames), probs, toJSON };
}
