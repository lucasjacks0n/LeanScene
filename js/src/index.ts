/** LeanScene: a 0.39M-parameter network for fast shot transition detection. This is the Node entry. */

import { resultFrom, streamProbs, type Result, type StreamOptions } from "./core.js";
import { decodeFrames, frameRate, type DecodeOptions } from "./decode.js";
import { loadModel, type Model, type ModelOptions } from "./model.js";
import type { RuleOptions } from "./rule.js";

export { FRAME_BYTES, FRAME_SIZE, frameFeatures, headProbs, type Result, type ResultJSON, type Runtime, type Span,
         type StreamOptions } from "./core.js";
export { decodeFrames, ffmpegPath, frameRate, type DecodeOptions } from "./decode.js";
export { loadModel, type Model, type ModelOptions } from "./model.js";
export { CUT_THRESHOLD, FADE_CLOSE, FADE_THRESHOLD, findShots, findTransitions, labelFrames,
         type RuleOptions, type Shot, type Transition } from "./rule.js";

/** A video path, or frames: FRAME_BYTES-long arrays of B, G, R bytes (see FRAME_SIZE). */
export type Source = string | AsyncIterable<Uint8Array> | Iterable<Uint8Array>;

export interface ProbsOptions extends StreamOptions, DecodeOptions {
  /** A loaded model to reuse across videos (loadModel()); loaded on each call otherwise. */
  model?: Model;
  /** How to load the model when `model` is not given. */
  modelOptions?: ModelOptions;
}

export interface DetectOptions extends ProbsOptions, RuleOptions {}

function framesOf(source: Source, options: DecodeOptions): AsyncIterable<Uint8Array> | Iterable<Uint8Array> {
  if (typeof source === "string") return decodeFrames(source, options);
  if (typeof Blob !== "undefined" && (source as unknown) instanceof Blob) {
    throw new Error("in Node, pass a file path or frames; File and Blob sources work in the browser build of this package");
  }
  return source;
}

/**
 * Per-frame probabilities of same / cut / fade: a Float32Array with three values per frame.
 *
 * `source` is a video path (decoded with ffmpeg) or an iterable of frames. Frames are
 * FRAME_SIZE x FRAME_SIZE, three bytes per pixel in B, G, R order, row by row.
 */
export async function frameProbs(source: Source, options: ProbsOptions = {}): Promise<Float32Array> {
  const frames = framesOf(source, options);
  const model = options.model ?? await loadModel(options.modelOptions);
  return streamProbs(model, frames, options);
}

/**
 * Find the shot transitions of a video.
 *
 * `source` is a video path, or an iterable of frames (see frameProbs). `cut` and `fade` are
 * probability thresholds: raise them for fewer false alarms, lower them to miss less.
 */
export async function detect(source: Source, options: DetectOptions = {}): Promise<Result> {
  const probs = await frameProbs(source, options);
  return resultFrom(probs, typeof source === "string" ? frameRate(source, options) : null, options);
}
