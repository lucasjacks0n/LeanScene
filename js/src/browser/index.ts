/**
 * LeanScene: a 0.39M-parameter network for fast shot transition detection. This is the browser
 * entry: the video is decoded with WebCodecs and the network runs on WebGPU or WebAssembly, all
 * in the page. Nothing is uploaded.
 */

import { resultFrom, streamProbs, type Result, type StreamOptions } from "../core.js";
import type { RuleOptions } from "../rule.js";
import { decodeFrames, type DecodeOptions, type VideoInfo } from "./decode.js";
import { loadModel, type Model, type ModelOptions } from "./model.js";

export { FRAME_BYTES, FRAME_SIZE, frameFeatures, headProbs, type Result, type ResultJSON, type Runtime, type Span,
         type StreamOptions } from "../core.js";
export { CUT_THRESHOLD, FADE_CLOSE, FADE_THRESHOLD, findShots, findTransitions, labelFrames,
         type RuleOptions, type Shot, type Transition } from "../rule.js";
export { decodeFrames, frameFrom, type DecodeOptions, type VideoInfo } from "./decode.js";
export { loadModel, type Model, type ModelOptions } from "./model.js";

/**
 * A video: a File or Blob (from a file input or a drop), a URL to fetch, the bytes of a file,
 * or a <video> element (its file is fetched). Or your own frames: an iterable of
 * FRAME_BYTES-long arrays of B, G, R bytes (see frameFrom).
 */
export type Source = Blob | string | URL | ArrayBuffer | Uint8Array | HTMLVideoElement
  | AsyncIterable<Uint8Array> | Iterable<Uint8Array>;

export interface ProbsOptions extends Omit<StreamOptions, "progress">, DecodeOptions {
  /** A loaded model to reuse across videos (loadModel()); loaded on each call otherwise. */
  model?: Model;
  /** How to load the model when `model` is not given. */
  modelOptions?: ModelOptions;
  /** Called after each batch with the frames done so far and, when the file says, how many it has. */
  progress?: (frames: number, total?: number) => void;
}

export interface DetectOptions extends ProbsOptions, RuleOptions {}

async function fetched(url: string | URL, signal?: AbortSignal): Promise<Blob> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`could not fetch ${url}: ${r.status}`);
  return r.blob();
}

/** The file behind a source, or null when the source is frames. */
async function fileOf(source: Source, signal?: AbortSignal): Promise<Blob | null> {
  if (source instanceof Blob) return source;
  if (typeof source === "string" || source instanceof URL) return fetched(source, signal);
  if (source instanceof ArrayBuffer || source instanceof Uint8Array) return new Blob([source as BlobPart]);
  if (typeof HTMLVideoElement !== "undefined" && source instanceof HTMLVideoElement) {
    const src = source.currentSrc || source.src;
    if (!src) throw new Error("the video element has no source yet");
    return fetched(src, signal);
  }
  return null;
}

async function probs(source: Source, options: ProbsOptions): Promise<{ probs: Float32Array; info: VideoInfo | null }> {
  const file = await fileOf(source, options.signal);
  let info: VideoInfo | null = null;
  const frames = file
    ? decodeFrames(file, { signal: options.signal, onTrack: i => { info = i; options.onTrack?.(i); } })
    : source as AsyncIterable<Uint8Array> | Iterable<Uint8Array>;
  const model = options.model ?? await loadModel(options.modelOptions);
  const progress = options.progress && ((n: number) => options.progress!(n, info?.frames ?? undefined));
  // Large batches suit a GPU; on WebAssembly small ones are faster, as on the CPU elsewhere.
  const batch = options.batch ?? (model.backend === "webgpu" ? 64 : 8);
  return { probs: await streamProbs(model, frames, { batch, window: options.window, progress }), info };
}

/**
 * Per-frame probabilities of same / cut / fade: a Float32Array with three values per frame.
 * `source` is a video (File, Blob, URL, bytes, <video> element) or an iterable of frames.
 */
export async function frameProbs(source: Source, options: ProbsOptions = {}): Promise<Float32Array> {
  return (await probs(source, options)).probs;
}

/**
 * Find the shot transitions of a video.
 *
 *     const { shots, transitions } = await detect(fileInput.files[0]);
 *
 * `cut` and `fade` are probability thresholds: raise them for fewer false alarms, lower them
 * to miss less.
 */
export async function detect(source: Source, options: DetectOptions = {}): Promise<Result> {
  const r = await probs(source, options);
  return resultFrom(r.probs, r.info ? (r.info as VideoInfo).fps : null, options);
}
