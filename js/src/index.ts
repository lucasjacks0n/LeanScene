/** LeanScene: a 0.39M-parameter network for fast shot transition detection. */

import { decodeFrames, frameRate, type DecodeOptions } from "./decode.js";
import { loadModel, streamProbs, type Model, type ModelOptions, type StreamOptions } from "./model.js";
import { findShots, findTransitions, labelFrames, type RuleOptions, type Shot, type Transition } from "./rule.js";

export { decodeFrames, ffmpegPath, frameRate, type DecodeOptions } from "./decode.js";
export { FRAME_BYTES, FRAME_SIZE, frameFeatures, headProbs, loadModel, type Model, type ModelOptions, type StreamOptions } from "./model.js";
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

/**
 * Per-frame probabilities of same / cut / fade: a Float32Array with three values per frame.
 *
 * `source` is a video path (decoded with ffmpeg) or an iterable of frames. Frames are
 * FRAME_SIZE x FRAME_SIZE, three bytes per pixel in B, G, R order, row by row.
 */
export async function frameProbs(source: Source, options: ProbsOptions = {}): Promise<Float32Array> {
  const model = options.model ?? await loadModel(options.modelOptions);
  const frames = typeof source === "string" ? decodeFrames(source, options) : source;
  return streamProbs(model, frames, options);
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

/**
 * Find the shot transitions of a video.
 *
 * `source` is a video path, or an iterable of frames (see frameProbs). `cut` and `fade` are
 * probability thresholds: raise them for fewer false alarms, lower them to miss less.
 */
export async function detect(source: Source, options: DetectOptions = {}): Promise<Result> {
  const probs = await frameProbs(source, options);
  const frames = probs.length / 3;
  const transitions = findTransitions(labelFrames(probs, options));
  const fps = typeof source === "string" ? frameRate(source, options) : null;
  return { frames, fps, transitions, shots: findShots(transitions, frames), probs, toJSON };
}
