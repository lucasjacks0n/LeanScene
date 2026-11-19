#!/usr/bin/env node
/** Command line: leanscene VIDEO [VIDEO ...] */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
import { parseArgs } from "node:util";
import { CUT_THRESHOLD, FADE_THRESHOLD, detect, loadModel, type ResultJSON } from "./index.js";

const HELP = `usage: leanscene [options] VIDEO [VIDEO ...]

Detect shot transitions (cuts and gradual transitions) in videos.

  -f, --format table|json|csv  table: one row per shot (default). json: shots and transitions. csv: one row per shot
  -o, --output FILE            write to this file instead of standard output
      --weights DIR            a directory with features.onnx, head.onnx and meta.json, to use other weights
      --threads N              threads for the model (default: ONNX Runtime decides)
      --cut X                  cut threshold (default ${CUT_THRESHOLD})
      --fade X                 gradual-transition threshold (default ${FADE_THRESHOLD})
      --probs FILE.npy         also save the per-frame probabilities (frames x 3: same, cut, fade); one video only
      --ffmpeg PATH            the ffmpeg program to decode with
  -q, --quiet                  no progress line (it goes to standard error, and is shown only in a terminal)
      --version
  -h, --help`;

const clock = (seconds: number) => {
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds - 3600 * h - 60 * m;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
};

function table(path: string, d: ResultJSON): string {
  const timed = d.fps !== null;
  const lines = [`${path}: ${d.frames} frames, ${timed ? `${d.fps!.toFixed(3)} fps` : "fps unknown"}, ${d.shots.length} shots, ${d.transitions.length} transitions`,
                 `${"shot".padStart(5)} ${"start".padStart(8)} ${"end".padStart(8)}` + (timed ? `  ${"start time".padStart(12)} ${"end time".padStart(12)}` : "")];
  d.shots.forEach((s, i) => lines.push(`${String(i + 1).padStart(5)} ${String(s.start).padStart(8)} ${String(s.end).padStart(8)}`
    + (timed ? `  ${clock(s.start_time!).padStart(12)} ${clock(s.end_time!).padStart(12)}` : "")));
  return lines.join("\n");
}

/** A NumPy .npy file holding a (rows, 3) float32 array. */
function npy(probs: Float32Array): Uint8Array {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${probs.length / 3}, 3), }`;
  header += " ".repeat(63 - ((10 + header.length) % 64)) + "\n";
  const out = new Uint8Array(10 + header.length + probs.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]);
  out.set(new TextEncoder().encode(header), 10);
  out.set(new Uint8Array(probs.buffer, probs.byteOffset, probs.byteLength), 10 + header.length);
  return out;
}

const csvCell = (v: string | number) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  const fail = (message: string) => {
    process.stderr.write(`leanscene: ${message}\n`);
    return 2;
  };
  let parsed;
  try {
    parsed = parseArgs({
      args: argv, allowPositionals: true,
      options: {
        format: { type: "string", short: "f", default: "table" }, output: { type: "string", short: "o" },
        weights: { type: "string" }, threads: { type: "string" }, cut: { type: "string" }, fade: { type: "string" },
        probs: { type: "string" }, ffmpeg: { type: "string" }, quiet: { type: "boolean", short: "q" },
        version: { type: "boolean" }, help: { type: "boolean", short: "h" },
      },
    });
  } catch (e) {
    return fail(`${(e as Error).message}\n${HELP}`);
  }
  const { values: o, positionals: videos } = parsed;
  if (o.help) return process.stdout.write(HELP + "\n"), 0;
  if (o.version) {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string };
    return process.stdout.write(`leanscene ${pkg.version}\n`), 0;
  }
  if (!videos.length) return fail(`no video given\n${HELP}`);
  if (!["table", "json", "csv"].includes(o.format!)) return fail(`--format must be table, json or csv`);
  if (o.probs && videos.length > 1) return fail("--probs takes a single video");
  const number = (name: "cut" | "fade" | "threads", fallback?: number) => {
    if (o[name] === undefined) return fallback;
    const v = Number(o[name]);
    if (!Number.isFinite(v)) throw new Error(`--${name} must be a number`);
    return v;
  };
  let cut: number, fade: number, threads: number | undefined;
  try {
    cut = number("cut", CUT_THRESHOLD)!;
    fade = number("fade", FADE_THRESHOLD)!;
    threads = number("threads");
  } catch (e) {
    return fail((e as Error).message);
  }

  const show = !o.quiet && process.stderr.isTTY;
  const model = await loadModel({ weights: o.weights, threads });
  const results = new Map<string, ResultJSON>();
  let failed = false;
  for (const [n, path] of videos.entries()) {
    if (!existsSync(path)) {
      process.stderr.write(`leanscene: ${path}: no such file\n`);
      failed = true;
      continue;
    }
    const label = basename(path) + (videos.length > 1 ? ` [${n + 1}/${videos.length}]` : "");
    const began = performance.now();
    let last = 0, width = 0;
    const line = (text: string, end = "") => {
      process.stderr.write(`\r${text}${" ".repeat(Math.max(0, width - text.length))}${end}`);
      width = text.length;
    };
    const rate = (frames: number) => Math.round(frames / Math.max((performance.now() - began) / 1000, 1e-6)).toLocaleString("en-US");
    const progress = show ? (frames: number) => {
      const now = performance.now();
      if (now - last < 200) return;
      last = now;
      line(`${label}: ${frames.toLocaleString("en-US")} frames, ${rate(frames)} frames/s`);
    } : undefined;
    try {
      const r = await detect(path, { model, cut, fade, progress, ffmpeg: o.ffmpeg });
      if (r.frames === 0) {
        if (show) line("", "\r");
        process.stderr.write(`leanscene: ${path}: no frames could be decoded\n`);
        failed = true;
        continue;
      }
      if (show) line(`${label}: ${r.frames.toLocaleString("en-US")} frames, ${rate(r.frames)} frames/s`, "\n");
      results.set(path, r.toJSON());
      if (o.probs) writeFileSync(o.probs, npy(r.probs));
    } catch (e) {
      if (show) line("", "\r");
      process.stderr.write(`leanscene: ${path}: ${(e as Error).message}\n`);
      failed = true;
    }
  }

  let text: string;
  if (o.format === "json") {
    text = JSON.stringify(videos.length > 1 ? Object.fromEntries(results) : [...results.values()][0] ?? {}, null, 1);
  } else if (o.format === "csv") {
    const rows = [["video", "shot", "start_frame", "end_frame", "start_time", "end_time"].join(",")];
    for (const [path, d] of results) {
      d.shots.forEach((s, i) => rows.push([path, i + 1, s.start, s.end, s.start_time ?? "", s.end_time ?? ""].map(csvCell).join(",")));
    }
    text = rows.join("\n");
  } else {
    text = [...results].map(([path, d]) => table(path, d)).join("\n\n");
  }
  if (o.output) writeFileSync(o.output, text + "\n");
  else if (text) process.stdout.write(text + "\n");
  return failed ? 1 : 0;
}

main().then(code => { process.exitCode = code; }, e => {
  process.stderr.write(`leanscene: ${(e as Error).message}\n`);
  process.exitCode = 1;
});
