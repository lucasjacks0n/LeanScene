/** The LeanScene network on onnxruntime-web: WebGPU when the browser has it, WebAssembly otherwise. */

import { checkMeta, FRAME_SIZE, type Meta, type Runtime } from "../core.js";

type Bytes = string | URL | ArrayBuffer | Uint8Array;

export interface ModelOptions {
  /** "webgpu" or "wasm" (the CPU). Default: WebGPU when the browser has it, else WebAssembly. */
  backend?: "webgpu" | "wasm";
  /**
   * Other weights than the bundled ones: features.onnx and head.onnx as URLs or bytes, and the
   * content of meta.json.
   */
  weights?: { features: Bytes; head: Bytes; meta: Meta };
  /**
   * Where ONNX Runtime's .wasm file is fetched from: a URL prefix ending in "/". By default it
   * comes from cdn.jsdelivr.net at the installed onnxruntime-web version, so nothing has to be
   * copied into your site. Pass your own prefix to serve it yourself, or false to leave ONNX
   * Runtime's setting (ort.env.wasm.wasmPaths) as it is.
   */
  wasmPaths?: string | false;
  /** WebAssembly threads. Default: up to 4 on a cross-origin isolated page, else 1 (a browser rule). */
  threads?: number;
}

export interface Model extends Runtime {
  /** What the network runs on. */
  readonly backend: "webgpu" | "wasm";
}

async function bytes(src: Bytes): Promise<Uint8Array> {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  const r = await fetch(src);
  if (!r.ok) throw new Error(`could not fetch ${src}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

const fromBase64 = (s: string) => Uint8Array.from(atob(s), c => c.charCodeAt(0));

/** Load the bundled weights (or `options.weights`) and start ONNX Runtime. */
export async function loadModel(options: ModelOptions = {}): Promise<Model> {
  const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator;
  const want = options.backend ?? (hasGpu ? "webgpu" : "wasm");
  // The WebGPU build of ONNX Runtime downloads a .wasm twice the size of the CPU-only one, so
  // it is loaded only when WebGPU is going to be used.
  const ort = want === "webgpu" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm");
  if (options.wasmPaths !== false && (options.wasmPaths || !ort.env.wasm.wasmPaths)) {
    ort.env.wasm.wasmPaths = options.wasmPaths || `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  }
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  ort.env.wasm.numThreads = options.threads ?? (isolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);

  let graphs: { features: Uint8Array; head: Uint8Array; meta: Meta };
  if (options.weights) {
    const [features, head] = await Promise.all([bytes(options.weights.features), bytes(options.weights.head)]);
    graphs = { features, head, meta: options.weights.meta };
  } else {
    const w = await import("./weights-data.js");
    graphs = { features: fromBase64(w.features), head: fromBase64(w.head), meta: w.meta };
  }
  const meta = checkMeta(graphs.meta);

  let backend = want;
  // Errors only: on WebGPU every session otherwise prints a warning that a few shape
  // operations stay on the CPU, which is how ONNX Runtime means it to be.
  const session = (graph: Uint8Array) => ort.InferenceSession.create(graph, { executionProviders: [backend], logSeverityLevel: 3 });
  let features;
  try {
    features = await session(graphs.features);
  } catch (e) {
    if (backend !== "webgpu" || options.backend === "webgpu") throw e;
    backend = "wasm";  // WebGPU is there but would not start (no adapter, a blocked GPU): use the CPU
    features = await session(graphs.features);
  }
  const head = await session(graphs.head);
  return {
    backend, dim: meta.dim, receptiveField: meta.receptiveField, checkpoint: meta.checkpoint, meta,
    async runFeatures(x, n) {
      return (await features.run({ x: new ort.Tensor("float32", x, [n, 3, FRAME_SIZE, FRAME_SIZE]) })).y.data as Float32Array;
    },
    async runHead(f, frames) {
      return (await head.run({ x: new ort.Tensor("float32", f, [1, frames, meta.dim]) })).y.data as Float32Array;
    },
  };
}
