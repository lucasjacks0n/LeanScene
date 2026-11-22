// The browser build, bundled and run in headless Chrome on the CPU (WebAssembly). Skipped when
// Chrome or ffmpeg is missing. Nothing is fetched from the network: ONNX Runtime's .wasm is
// served from node_modules through the wasmPaths option.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { createReadStream, existsSync, mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join } from "node:path";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";
import { ffmpegPath } from "../dist/index.js";

const root = fileURLToPath(new URL("..", import.meta.url));
const chrome = [process.env.CHROME_PATH, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"].find(p => p && existsSync(p));
const hasFfmpeg = spawnSync(ffmpegPath(), ["-version"]).status === 0;
const skip = !chrome ? "Chrome is not installed (set CHROME_PATH)" : !hasFfmpeg ? "ffmpeg is not installed" : false;

let dir, server, browser, page;
before(async () => {
  if (skip) return;
  const { build } = await import("esbuild");
  const puppeteer = (await import("puppeteer-core")).default;
  dir = mkdtempSync(join(tmpdir(), "leanscene-browser-"));
  // three plain shots of 40 frames at 25 fps: cuts at frames 40 and 80
  const part = src => ["-f", "lavfi", "-t", "1.6", "-i", `${src}=size=320x240:rate=25`];
  execFileSync(ffmpegPath(), ["-v", "error", ...part("testsrc2"), ...part("smptebars"), ...part("mandelbrot"),
    "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p", "-c:v", "libx264", "-y", join(dir, "three-shots.mp4")]);
  // A larger video, read in several 1 MB chunks (about 5 MB): three 4-second shots with noise so
  // that it does not compress away, cuts at frames 100 and 200. ffmpeg puts the index (moov) at
  // the end, as many cameras do; the second copy has it at the start ("faststart").
  const long = src => ["-f", "lavfi", "-t", "4", "-i", `${src}=size=640x360:rate=25`];
  execFileSync(ffmpegPath(), ["-v", "error", ...long("testsrc2"), ...long("smptebars"), ...long("mandelbrot"),
    "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,noise=alls=20:allf=t,format=yuv420p",
    "-c:v", "libx264", "-crf", "28", "-y", join(dir, "index-at-end.mp4")]);
  execFileSync(ffmpegPath(), ["-v", "error", "-i", join(dir, "index-at-end.mp4"), "-c", "copy", "-movflags", "+faststart",
    "-y", join(dir, "index-at-start.mp4")]);
  writeFileSync(join(dir, "app.js"), `import * as leanscene from ${JSON.stringify(join(root, "dist/browser/index.js"))};\nwindow.leanscene = leanscene;\n`);
  await build({ entryPoints: [join(dir, "app.js")], bundle: true, format: "esm", splitting: true, platform: "browser",
                outdir: join(dir, "out"), logLevel: "silent" });
  const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".mp4": "video/mp4", ".wasm": "application/wasm" };
  server = createServer((req, res) => {
    const url = req.url.split("?")[0];
    if (url === "/") {
      return res.writeHead(200, { "content-type": "text/html" })
        .end('<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><script type="module" src="/out/app.js"></script>');
    }
    const path = url.startsWith("/ort/") ? join(root, "node_modules/onnxruntime-web/dist", url.slice(5)) : join(dir, url);
    if (!existsSync(path)) return res.writeHead(404).end();
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream", "content-length": statSync(path).size });
    createReadStream(path).pipe(res);
  }).listen(0);
  browser = await puppeteer.launch({ executablePath: chrome, headless: "new", args: ["--no-sandbox", "--enable-unsafe-webgpu"] });
  page = await open();
  await page.evaluate(async () => { window.model = await leanscene.loadModel({ backend: "wasm", wasmPaths: "/ort/" }); });
});

/** A fresh page with the bundle loaded. `before` runs in it before the bundle does. */
async function open(before) {
  const p = await browser.newPage();
  if (before) await p.evaluateOnNewDocument(before);
  await p.goto(`http://127.0.0.1:${server.address().port}/`);
  await p.waitForFunction("window.leanscene !== undefined", { timeout: 60000 });
  return p;
}
after(async () => {
  await browser?.close();
  server?.close();
  if (dir) rmSync(dir, { recursive: true, force: true });
});

test("browser: the bundled weights load on WebAssembly", { skip }, async () => {
  assert.deepEqual(await page.evaluate(() => [model.backend, model.dim, model.receptiveField]), ["wasm", 896, 63]);
});

test("browser: detect on a fetched MP4 finds both cuts, the frame rate and the frame count", { skip }, async () => {
  const r = await page.evaluate(async () => {
    const seen = [];
    const result = await leanscene.detect("/three-shots.mp4", { model, progress: (n, total) => seen.push([n, total]) });
    return { json: result.toJSON(), probs: result.probs.length, last: seen[seen.length - 1] };
  });
  assert.equal(r.json.frames, 120);
  assert.equal(r.json.fps, 25);
  assert.deepEqual(r.json.transitions.map(t => [t.start, t.kind]), [[40, "cut"], [80, "cut"]]);
  assert.deepEqual(r.json.shots[1], { start: 40, end: 79, start_time: 1.6, end_time: 3.2 });
  assert.equal(r.probs, 360);
  assert.deepEqual(r.last, [120, 120]);
});

test("browser: a File gives the same result as its URL", { skip }, async () => {
  const same = await page.evaluate(async () => {
    const blob = await (await fetch("/three-shots.mp4")).blob();
    const a = await leanscene.detect(new File([blob], "three-shots.mp4"), { model });
    const b = await leanscene.detect("/three-shots.mp4", { model });
    return JSON.stringify(a) === JSON.stringify(b) && a.frames === 120;
  });
  assert.ok(same);
});

test("browser: your own frames, made with frameFrom", { skip }, async () => {
  const r = await page.evaluate(async () => {
    const c = new OffscreenCanvas(320, 240), x = c.getContext("2d"), frames = [];
    for (let k = 0; k < 80; k++) {
      x.fillStyle = k < 40 ? "#204080" : "#c0a020";
      x.fillRect(0, 0, 320, 240);
      x.fillStyle = k < 40 ? "#e0e0e0" : "#101010";
      x.fillRect(20 + k, 40, 90, 120);  // something that moves, so a shot is not a still image
      frames.push(leanscene.frameFrom(c));
    }
    const result = await leanscene.detect(frames, { model });
    return { bytes: frames[0].length, fps: result.fps, transitions: result.transitions };
  });
  assert.equal(r.bytes, 30000);
  assert.equal(r.fps, null);
  assert.deepEqual(r.transitions, [{ start: 40, end: 40, kind: "cut" }]);
});

test("browser: a file that is not a video is an error, and so is a missing one", { skip }, async () => {
  const errors = await page.evaluate(async () => {
    const message = async source => { try { await leanscene.detect(source, { model }); return "no error"; } catch (e) { return e.message; } };
    return [await message(new Blob(["this is not a video"])), await message("/nope.mp4")];
  });
  assert.match(errors[0], /MP4 or MOV|MP4\/MOV/);
  assert.match(errors[1], /could not fetch/);
});

test("browser: a file read in several chunks, with its index at the end or at the start", { skip }, async () => {
  const r = await page.evaluate(async () => {
    const run = async url => {
      const x = await leanscene.detect(url, { model });
      return { frames: x.frames, fps: x.fps, cuts: x.transitions.map(t => [t.start, t.kind]), probs: Array.from(x.probs) };
    };
    const size = (await (await fetch("/index-at-end.mp4")).blob()).size;
    return { size, end: await run("/index-at-end.mp4"), start: await run("/index-at-start.mp4") };
  });
  assert.ok(r.size > 3 * 2 ** 20, `the fixture is ${r.size} bytes; it must span several 1 MB chunks`);
  for (const x of [r.end, r.start]) {
    assert.equal(x.frames, 300);
    assert.equal(x.fps, 25);
    assert.deepEqual(x.cuts, [[100, "cut"], [200, "cut"]]);
  }
  assert.deepEqual(r.end.probs, r.start.probs);  // the same frames, whichever way the file is laid out
});

test("browser: the default backend (WebGPU when it starts, else WebAssembly) gives the same result", { skip }, async () => {
  const p = await open();
  try {
    const r = await p.evaluate(async () => {
      const m = await leanscene.loadModel({ wasmPaths: "/ort/" });
      const x = await leanscene.detect("/three-shots.mp4", { model: m });
      return { backend: m.backend, gpu: "gpu" in navigator, json: x.toJSON() };
    });
    assert.ok(["webgpu", "wasm"].includes(r.backend));
    if (!r.gpu) assert.equal(r.backend, "wasm");
    assert.deepEqual(r.json.transitions.map(t => [t.start, t.kind]), [[40, "cut"], [80, "cut"]]);
  } finally {
    await p.close();
  }
});

test("browser: when WebGPU is present but will not start, the model falls back to WebAssembly", { skip }, async () => {
  // a browser that has WebGPU but no usable GPU: requestAdapter() resolves to null
  const p = await open(() => {
    if (!("gpu" in navigator)) Object.defineProperty(navigator, "gpu", { value: {}, configurable: true });
    navigator.gpu.requestAdapter = async () => null;
  });
  try {
    const r = await p.evaluate(async () => {
      const m = await leanscene.loadModel({ wasmPaths: "/ort/" });
      const x = await leanscene.detect("/three-shots.mp4", { model: m });
      const forced = await leanscene.loadModel({ backend: "webgpu", wasmPaths: "/ort/" }).then(() => "loaded", e => "error");
      return { backend: m.backend, cuts: x.transitions.map(t => t.start), forced };
    });
    assert.equal(r.backend, "wasm");
    assert.deepEqual(r.cuts, [40, 80]);
    assert.equal(r.forced, "error");  // asked for WebGPU explicitly: no silent fallback
  } finally {
    await p.close();
  }
});
