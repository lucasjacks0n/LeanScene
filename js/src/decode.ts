/** Decode a video file with ffmpeg into the frames the network reads. */

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { FRAME_BYTES, FRAME_SIZE } from "./model.js";

export interface DecodeOptions {
  /** The ffmpeg program (default: $LEANSCENE_FFMPEG, then the `ffmpeg-static` package if installed, then "ffmpeg" on PATH). */
  ffmpeg?: string;
  /** Stop the decoder early when this signal aborts. */
  signal?: AbortSignal;
}

// The Python package resizes with OpenCV's bilinear filter, which reads only the pixels nearest
// each sample point. Of ffmpeg's scalers, point sampling came closest to it in our comparison
// (js/README.md); its smoothing filters average over the whole area a pixel covers.
const SCALER = "neighbor";

let found: string | undefined;

/** The ffmpeg program used when none is given. */
export function ffmpegPath(): string {
  if (found) return found;
  if (process.env.LEANSCENE_FFMPEG) return (found = process.env.LEANSCENE_FFMPEG);
  try {
    const p = createRequire(import.meta.url)("ffmpeg-static") as string | null;
    if (p && existsSync(p)) return (found = p);
  } catch {
    // not installed: fall through to PATH
  }
  return (found = "ffmpeg");
}

// "Keep every frame as it is" is -fps_mode passthrough from ffmpeg 5.1, and -vsync passthrough before.
const passthrough = new Map<string, string[]>();
function passthroughArgs(program: string): string[] {
  let args = passthrough.get(program);
  if (!args) {
    const m = /^ffmpeg version n?(\d+)\.(\d+)/.exec(spawnSync(program, ["-version"], { encoding: "utf8" }).stdout ?? "");
    const old = m !== null && (+m[1] < 5 || (+m[1] === 5 && +m[2] < 1));
    passthrough.set(program, args = old ? ["-vsync", "passthrough"] : ["-fps_mode", "passthrough"]);
  }
  return args;
}

const missing = (program: string) =>
  new Error(`cannot run ${program}: install ffmpeg (https://ffmpeg.org), or the ffmpeg-static package, or pass the ffmpeg option`);

/**
 * Every frame of the first video stream, in display order, as FRAME_SIZE x FRAME_SIZE pixels of
 * three bytes each (B, G, R), row by row. No frame is duplicated or dropped to even out timing,
 * so the n-th frame yielded is the n-th frame a frame-by-frame reader such as OpenCV returns.
 */
export async function* decodeFrames(path: string, options: DecodeOptions = {}): AsyncGenerator<Uint8Array, void, void> {
  if (!existsSync(path)) throw Object.assign(new Error(`no such file: ${path}`), { code: "ENOENT" });
  const program = options.ffmpeg ?? ffmpegPath();
  const args = ["-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", `scale=${FRAME_SIZE}:${FRAME_SIZE}:flags=${SCALER}`, ...passthroughArgs(program),
                "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"];
  const child = spawn(program, args, { stdio: ["ignore", "pipe", "pipe"], signal: options.signal });
  let stderr = "";
  child.stderr.setEncoding("utf8").on("data", (s: string) => { if (stderr.length < 4000) stderr += s; });
  let failure: Error | undefined;
  const closed = new Promise<number | null>(resolve => {
    child.on("error", e => {
      failure = (e as NodeJS.ErrnoException).code === "ENOENT" ? missing(program) : e;
      resolve(null);
    });
    child.on("close", code => resolve(code));
  });

  let frame = new Uint8Array(FRAME_BYTES), filled = 0, finished = false;
  try {
    for await (const chunk of child.stdout as AsyncIterable<Buffer>) {
      let at = 0;
      while (at < chunk.length) {
        const n = Math.min(FRAME_BYTES - filled, chunk.length - at);
        frame.set(chunk.subarray(at, at + n), filled);
        filled += n;
        at += n;
        if (filled === FRAME_BYTES) {
          yield frame;
          frame = new Uint8Array(FRAME_BYTES);
          filled = 0;
        }
      }
    }
    finished = true;
  } catch (e) {
    if (!failure) failure = e as Error;
  } finally {
    if (!finished) child.kill();  // the consumer stopped early, or the stream broke
  }
  const code = await closed;
  if (failure) throw failure;
  if (finished && code !== 0) throw new Error(`cannot decode ${path}: ${stderr.trim() || `ffmpeg exited with code ${code}`}`);
}

/**
 * The first video stream's average frame rate, as ffprobe reports it; null when ffprobe is
 * missing or the file does not say.
 */
export function frameRate(path: string, options: { ffprobe?: string; ffmpeg?: string } = {}): number | null {
  const ffmpeg = options.ffmpeg ?? ffmpegPath();
  const program = options.ffprobe ?? process.env.LEANSCENE_FFPROBE ?? ffmpeg.replace(/ffmpeg(\.exe)?$/i, "ffprobe$1");
  const r = spawnSync(program, ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                                "-of", "default=noprint_wrappers=1:nokey=0", path], { encoding: "utf8" });
  if (r.error || r.status !== 0) return null;
  const rate = (key: string) => {
    const m = new RegExp(`^${key}=(\\d+)/(\\d+)`, "m").exec(r.stdout);
    return m && +m[2] > 0 && +m[1] > 0 ? +m[1] / +m[2] : null;
  };
  return rate("avg_frame_rate") ?? rate("r_frame_rate");
}
