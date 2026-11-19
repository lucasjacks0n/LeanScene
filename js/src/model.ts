/** The LeanScene network as two ONNX graphs, and the streaming pass that runs it over a video. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";

/** Frames are FRAME_SIZE x FRAME_SIZE before the network sees them. */
export const FRAME_SIZE = 100;
/** Bytes in one frame: FRAME_SIZE x FRAME_SIZE pixels, three bytes each, in B, G, R order. */
export const FRAME_BYTES = FRAME_SIZE * FRAME_SIZE * 3;

const PIXELS = FRAME_SIZE * FRAME_SIZE;

interface Meta {
  checkpoint: string;
  dim: number;
  feat: number;
  size: number;
  histBins: number;
  receptiveField: number;
}

export interface ModelOptions {
  /** Directory holding features.onnx, head.onnx and meta.json, to use weights other than the bundled ones. */
  weights?: string;
  /** Threads for one inference call (default: ONNX Runtime decides). */
  threads?: number;
  /** ONNX Runtime execution providers, e.g. ["cuda"] or ["coreml"] (default: the CPU). */
  executionProviders?: string[];
}

export interface Model {
  readonly features: ort.InferenceSession;
  readonly head: ort.InferenceSession;
  /** Values per frame the temporal head reads. */
  readonly dim: number;
  /** Number of frames that can influence one frame's label. */
  readonly receptiveField: number;
  /** Name of the checkpoint the graphs were exported from. */
  readonly checkpoint: string;
  /** @internal */
  readonly meta: Meta;
}

/** Load the bundled weights, or the ones in `options.weights`. */
export async function loadModel(options: ModelOptions = {}): Promise<Model> {
  const dir = options.weights ?? fileURLToPath(new URL("../weights/", import.meta.url));
  const at = (name: string) => `${dir.replace(/[\\/]$/, "")}/${name}`;
  const meta = JSON.parse(readFileSync(at("meta.json"), "utf8")) as Meta;
  if (meta.size !== FRAME_SIZE) throw new Error(`these weights expect ${meta.size}x${meta.size} frames, not ${FRAME_SIZE}`);
  const so: ort.InferenceSession.SessionOptions = {};
  if (options.threads) so.intraOpNumThreads = options.threads;
  if (options.executionProviders) so.executionProviders = options.executionProviders;
  const [features, head] = await Promise.all([
    ort.InferenceSession.create(at("features.onnx"), so),
    ort.InferenceSession.create(at("head.onnx"), so),
  ]);
  return { features, head, dim: meta.dim, receptiveField: meta.receptiveField, checkpoint: meta.checkpoint, meta };
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
export async function frameFeatures(model: Model, frames: Uint8Array[]): Promise<Float32Array> {
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
  const y = (await model.features.run({ x: new ort.Tensor("float32", x, [n, 3, FRAME_SIZE, FRAME_SIZE]) })).y.data as Float32Array;
  const rest = dim - feat - nb;  // the graph returns [CNN mean | thumbnail | grid]; the histogram goes after the mean
  for (let k = 0; k < n; k++) {
    const src = k * (dim - nb), dst = k * dim;
    out.set(y.subarray(src, src + feat), dst);
    out.set(y.subarray(src + feat, src + feat + rest), dst + feat + nb);
  }
  return out;
}

/** Probabilities of same / cut / fade for a run of frames, from their features: three values per frame. */
export async function headProbs(model: Model, features: Float32Array, frames: number): Promise<Float32Array> {
  const y = await model.head.run({ x: new ort.Tensor("float32", features, [1, frames, model.dim]) });
  return y.y.data as Float32Array;
}

export interface StreamOptions {
  /** Frames encoded per call (default 8). */
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
export async function streamProbs(model: Model, frames: AsyncIterable<Uint8Array> | Iterable<Uint8Array>,
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
