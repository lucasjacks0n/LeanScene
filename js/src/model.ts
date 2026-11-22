/** The LeanScene network on onnxruntime-node. */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { checkMeta, FRAME_SIZE, type Meta, type Runtime } from "./core.js";

export { FRAME_BYTES, FRAME_SIZE, frameFeatures, headProbs, streamProbs, type StreamOptions } from "./core.js";

export interface ModelOptions {
  /** Directory holding features.onnx, head.onnx and meta.json, to use weights other than the bundled ones. */
  weights?: string;
  /** Threads for one inference call (default: ONNX Runtime decides). */
  threads?: number;
  /** ONNX Runtime execution providers, e.g. ["cuda"] or ["coreml"] (default: the CPU). */
  executionProviders?: string[];
}

export interface Model extends Runtime {
  readonly features: ort.InferenceSession;
  readonly head: ort.InferenceSession;
}

/** Load the bundled weights, or the ones in `options.weights`. */
export async function loadModel(options: ModelOptions = {}): Promise<Model> {
  const dir = options.weights ?? fileURLToPath(new URL("../weights/", import.meta.url));
  const at = (name: string) => `${dir.replace(/[\\/]$/, "")}/${name}`;
  const meta = checkMeta(JSON.parse(readFileSync(at("meta.json"), "utf8")) as Meta);
  const so: ort.InferenceSession.SessionOptions = {};
  if (options.threads) so.intraOpNumThreads = options.threads;
  if (options.executionProviders) so.executionProviders = options.executionProviders;
  const [features, head] = await Promise.all([
    ort.InferenceSession.create(at("features.onnx"), so),
    ort.InferenceSession.create(at("head.onnx"), so),
  ]);
  return {
    features, head, dim: meta.dim, receptiveField: meta.receptiveField, checkpoint: meta.checkpoint, meta,
    async runFeatures(x, n) {
      return (await features.run({ x: new ort.Tensor("float32", x, [n, 3, FRAME_SIZE, FRAME_SIZE]) })).y.data as Float32Array;
    },
    async runHead(f, frames) {
      return (await head.run({ x: new ort.Tensor("float32", f, [1, frames, meta.dim]) })).y.data as Float32Array;
    },
  };
}
