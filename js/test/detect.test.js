// The model on frames, the ffmpeg decoder, and the command line.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { FRAME_BYTES, FRAME_SIZE, decodeFrames, detect, ffmpegPath, frameProbs, loadModel } from "../dist/index.js";

const CLI = new URL("../dist/cli.js", import.meta.url).pathname;
const hasFfmpeg = spawnSync(ffmpegPath(), ["-version"]).status === 0;
const needsFfmpeg = { skip: hasFfmpeg ? false : "ffmpeg is not installed" };

/** n frames of one smooth pattern drifting sideways; `seed` picks the pattern. */
function shot(seed, n) {
  const frames = [];
  for (let k = 0; k < n; k++) {
    const f = new Uint8Array(FRAME_BYTES);
    for (let y = 0; y < FRAME_SIZE; y++) for (let x = 0; x < FRAME_SIZE; x++) {
      const i = (y * FRAME_SIZE + x) * 3, u = (x + k) / 17 + seed, v = y / 23 + seed * 2;
      f[i] = 128 + 100 * Math.sin(u * seed);
      f[i + 1] = 128 + 100 * Math.cos(v + seed);
      f[i + 2] = 128 + 100 * Math.sin(u + v);
    }
    frames.push(f);
  }
  return frames;
}

let model, dir, video;
before(async () => {
  model = await loadModel();
  dir = mkdtempSync(join(tmpdir(), "leanscene-test-"));
  video = join(dir, "three-shots.mp4");
  if (hasFfmpeg) {
    // three plain shots of 40 frames at 25 fps: cuts at frames 40 and 80
    const part = src => ["-f", "lavfi", "-t", "1.6", "-i", `${src}=size=320x240:rate=25`];
    execFileSync(ffmpegPath(), ["-v", "error", ...part("testsrc2"), ...part("smptebars"), ...part("mandelbrot"),
      "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p", "-y", video]);
  }
});
after(() => rmSync(dir, { recursive: true, force: true }));

test("the bundled weights are the paper's model", () => {
  assert.equal(model.dim, 896);
  assert.equal(model.receptiveField, 63);
});

test("a hard cut between two shots is found on the first frame of the new shot", async () => {
  const r = await detect([...shot(1, 40), ...shot(2, 40)], { model });
  assert.equal(r.frames, 80);
  assert.equal(r.fps, null);
  assert.deepEqual(r.transitions, [{ start: 40, end: 40, kind: "cut" }]);
  assert.deepEqual(r.shots, [[0, 39], [40, 79]]);
  assert.equal(r.probs.length, 240);
  assert.deepEqual(JSON.parse(JSON.stringify(r)).shots, [{ start: 0, end: 39 }, { start: 40, end: 79 }]);
});

test("streaming in small windows equals one pass", async () => {
  const frames = [...shot(1, 150), ...shot(2, 170), ...shot(3, 130)];
  const whole = await frameProbs(frames, { model, window: 100000 });
  const pieces = await frameProbs(frames, { model, window: 64, batch: 7 });
  assert.equal(pieces.length, whole.length);
  let worst = 0;
  for (let i = 0; i < whole.length; i++) worst = Math.max(worst, Math.abs(whole[i] - pieces[i]));
  assert.ok(worst < 1e-4, `largest difference ${worst}`);
});

test("a frame of the wrong size is refused", async () => {
  await assert.rejects(frameProbs([new Uint8Array(10)], { model }), /30000 bytes/);
});

test("no frames gives an empty result", async () => {
  const r = await detect([], { model });
  assert.deepEqual([r.frames, r.transitions, r.shots], [0, [], []]);
});

test("ffmpeg decodes every frame once, in order", needsFfmpeg, async () => {
  let n = 0;
  for await (const f of decodeFrames(video)) {
    assert.equal(f.length, FRAME_BYTES);
    n++;
  }
  assert.equal(n, 120);
});

test("detect on a file finds both cuts and reports the frame rate", needsFfmpeg, async () => {
  const r = await detect(video, { model });
  assert.equal(r.frames, 120);
  assert.equal(r.fps, 25);
  assert.deepEqual(r.transitions.map(t => [t.start, t.kind]), [[40, "cut"], [80, "cut"]]);
  const j = r.toJSON();
  assert.deepEqual(j.shots[1], { start: 40, end: 79, start_time: 1.6, end_time: 3.2 });
});

test("a missing file and a file that is not a video are errors", needsFfmpeg, async () => {
  await assert.rejects(detect(join(dir, "nope.mp4"), { model }), /no such file/);
  await assert.rejects(detect(CLI, { model }), /cannot decode/);
});

test("a missing ffmpeg is reported as such", async () => {
  await assert.rejects(detect(CLI, { model, ffmpeg: join(dir, "no-ffmpeg-here") }), /install ffmpeg/);
});

test("command line: json, csv and --probs", needsFfmpeg, () => {
  const run = (...args) => spawnSync(process.execPath, [CLI, ...args], { encoding: "utf8" });
  const j = JSON.parse(run(video, "-f", "json").stdout);
  assert.equal(j.frames, 120);
  assert.deepEqual(j.transitions.map(t => t.start), [40, 80]);
  const npy = join(dir, "p.npy");
  const csv = run(video, "-f", "csv", "--probs", npy).stdout.trim().split("\n");
  assert.equal(csv[0], "video,shot,start_frame,end_frame,start_time,end_time");
  assert.equal(csv.length, 4);
  const bytes = readFileSync(npy);
  assert.equal(bytes.subarray(1, 6).toString(), "NUMPY");
  assert.equal((bytes.length - 10 - bytes.readUInt16LE(8)) / 4, 360);
  const bad = run(join(dir, "nope.mp4"));
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /no such file/);
});
