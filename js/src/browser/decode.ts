/** Decode a video in the browser (mp4box + WebCodecs) into the frames the network reads. */

import { createFile, DataStream, Endianness, MP4BoxBuffer, type Movie, type Sample } from "mp4box";
import { FRAME_BYTES, FRAME_SIZE } from "../core.js";

/**
 * The first video track. `frames` and `fps` start as what the file's header says, which a
 * fragmented file (one written for streaming) does not know: they are null there. Both are
 * set to what was actually read once decoding ends.
 */
export interface VideoInfo {
  codec: string;
  width: number;
  height: number;
  /** Frames in the track, or null when not known yet. */
  frames: number | null;
  /** Average frames per second, or null when not known. */
  fps: number | null;
}

export interface DecodeOptions {
  /** Stop decoding when this signal aborts. */
  signal?: AbortSignal;
  /** Called once, when the video track has been read from the file. */
  onTrack?: (info: VideoInfo) => void;
}

const TILES = 8;  // frames are resized into an atlas of TILES x TILES tiles and read back together
const CHUNK = 1 << 20;

function atlasContext(): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D {
  const side = TILES * FRAME_SIZE;
  // willReadFrequently keeps the pixels in CPU memory: reading them back from a GPU canvas was
  // the slowest step of the demo page (415 against 1,300 frames/s on 720p, measured there).
  const ctx = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(side, side).getContext("2d", { willReadFrequently: true })
    : Object.assign(document.createElement("canvas"), { width: side, height: side }).getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("could not create a 2D canvas to resize frames on");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "low";  // bilinear, the closest to the Python package's resize
  return ctx;
}

/** `n` tiles of an RGBA atlas as separate frames of B, G, R bytes. */
function tiles(rgba: Uint8ClampedArray, n: number, out: Uint8Array[]): void {
  const width = TILES * FRAME_SIZE;
  for (let k = 0; k < n; k++) {
    const f = new Uint8Array(FRAME_BYTES), ox = (k % TILES) * FRAME_SIZE, oy = Math.floor(k / TILES) * FRAME_SIZE;
    for (let y = 0, p = 0; y < FRAME_SIZE; y++) {
      for (let q = ((oy + y) * width + ox) * 4, end = q + 4 * FRAME_SIZE; q < end; q += 4, p += 3) {
        f[p] = rgba[q + 2];
        f[p + 1] = rgba[q + 1];
        f[p + 2] = rgba[q];
      }
    }
    out.push(f);
  }
}

// The decoder is handed the codec's configuration record: the avcC / hvcC / vpcC / av1C box without its header.
function codecDescription(trak: any): Uint8Array | undefined {
  for (const entry of trak.mdia.minf.stbl.stsd.entries) {
    const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
    if (box) {
      const ds = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
      box.write(ds);
      return new Uint8Array(ds.buffer, 8);
    }
  }
}

/**
 * Every frame of the first video track of an MP4 or MOV file, in display order, as
 * FRAME_SIZE x FRAME_SIZE pixels of three bytes each (B, G, R), row by row.
 */
export async function* decodeFrames(file: Blob, options: DecodeOptions = {}): AsyncGenerator<Uint8Array, void, void> {
  if (typeof VideoDecoder === "undefined") {
    throw new Error("this browser has no WebCodecs video decoder; use a current Chrome, Edge, Safari or Firefox, or pass your own frames");
  }
  const atlas = atlasContext();
  const ready: Uint8Array[] = [];  // decoded frames not yet handed out
  let at = 0, fill = 0, done = false, stopped = false, failure: Error | undefined;
  let wake: (() => void) | undefined;
  const notify = () => { wake?.(); wake = undefined; };
  const fail = (e: unknown) => {
    if (!failure) failure = e instanceof Error ? e : new Error(String(e));
    notify();
  };
  const readAtlas = () => {
    if (!fill) return;
    tiles(atlas.getImageData(0, 0, TILES * FRAME_SIZE, Math.ceil(fill / TILES) * FRAME_SIZE).data, fill, ready);
    fill = 0;
    notify();
  };

  const mp4 = createFile();
  let decoder: VideoDecoder | undefined;
  let info: VideoInfo | undefined, count = 0, ticks = 0, timescale = 0;
  mp4.onError = (_module: string, message: string) => fail(new Error(`could not read this file as MP4/MOV (${message})`));
  mp4.onReady = (info_: Movie) => {
    const t = info_.videoTracks[0];
    if (!t) return fail(new Error("no video track found in this file"));
    decoder = new VideoDecoder({
      output: frame => {
        try {
          if (!stopped && !failure) {
            atlas.drawImage(frame, (fill % TILES) * FRAME_SIZE, Math.floor(fill / TILES) * FRAME_SIZE, FRAME_SIZE, FRAME_SIZE);
            if (++fill === TILES * TILES) readAtlas();
          }
        } catch (e) {
          fail(e);
        } finally {
          frame.close();
        }
      },
      error: e => fail(new Error(`this browser could not decode the video (codec ${t.codec}): ${e.message}`)),
    });
    try {
      decoder.configure({ codec: t.codec, codedWidth: t.video!.width, codedHeight: t.video!.height,
                          description: codecDescription(mp4.getTrackById(t.id)) });
    } catch (e) {
      return fail(new Error(`unsupported video codec ${t.codec}: ${(e as Error).message}`));
    }
    const seconds = t.duration / t.timescale, known = !info_.isFragmented && t.nb_samples > 0;
    info = { codec: t.codec, width: t.video!.width, height: t.video!.height, frames: known ? t.nb_samples : null,
             fps: known && seconds > 0 ? t.nb_samples / seconds : null };
    options.onTrack?.(info);
    mp4.setExtractionOptions(t.id, null, { nbSamples: 100 });
    mp4.start();
  };
  mp4.onSamples = (id: number, _user: unknown, samples: Sample[]) => {
    if (stopped || failure || !decoder) return;
    for (const s of samples) {
      count++;
      ticks += s.duration;
      timescale = s.timescale;
      decoder.decode(new EncodedVideoChunk({ type: s.is_sync ? "key" : "delta", timestamp: 1e6 * s.cts / s.timescale,
                                             duration: 1e6 * s.duration / s.timescale, data: s.data! }));
    }
    mp4.releaseUsedSamples(id, samples[samples.length - 1].number);
  };

  const idle = () => new Promise<void>(r => setTimeout(r, 4));
  const pump = (async () => {
    try {
      for (let off = 0; off < file.size && !stopped && !failure; off += CHUNK) {
        mp4.appendBuffer(MP4BoxBuffer.fromArrayBuffer(await file.slice(off, off + CHUNK).arrayBuffer(), off));
        // wait while the decoder or the consumer is behind, so a long video is never held in memory
        while (!stopped && !failure && ((decoder && decoder.decodeQueueSize > 40) || ready.length - at > 4 * TILES * TILES)) await idle();
      }
      if (stopped || failure) return;
      mp4.flush();
      if (!decoder) throw new Error("this does not look like an MP4 or MOV video (WebM, MKV and AVI are not read)");
      await decoder.flush();
      readAtlas();
      if (info) {  // the measured values: every sample's duration, as a frame-by-frame reader would see it
        info.frames = count;
        if (ticks > 0 && timescale > 0) info.fps = count / (ticks / timescale);
      }
      done = true;
      notify();
    } catch (e) {
      fail(e);
    }
  })();
  const abort = () => fail(options.signal!.reason ?? new DOMException("aborted", "AbortError"));
  if (options.signal?.aborted) abort();
  options.signal?.addEventListener("abort", abort);

  try {
    for (;;) {
      if (failure) throw failure;
      if (at < ready.length) {
        yield ready[at++];
        if (at > 1024) {  // drop the frames already handed out
          ready.splice(0, at);
          at = 0;
        }
      } else if (done) {
        return;
      } else {
        await new Promise<void>(r => { wake = r; });
      }
    }
  } finally {
    stopped = true;
    options.signal?.removeEventListener("abort", abort);
    await pump;
    try {
      if (decoder && decoder.state !== "closed") decoder.close();
    } catch {
      // already closed by an error
    }
  }
}

let one: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | undefined;

/**
 * One frame for the network from anything a canvas can draw: a video element at its current
 * time, an image, a canvas, an ImageBitmap or a VideoFrame. For feeding detect() your own frames.
 */
export function frameFrom(image: CanvasImageSource): Uint8Array {
  one ??= atlasContext();
  one.drawImage(image, 0, 0, FRAME_SIZE, FRAME_SIZE);
  const out: Uint8Array[] = [];
  tiles(one.getImageData(0, 0, TILES * FRAME_SIZE, FRAME_SIZE).data, 1, out);
  return out[0];
}
