// Compare the browser build with the Node build on real videos.
//
//   node scripts/browser_parity.mjs [webgpu|wasm|auto] VIDEO.mp4 [VIDEO.mp4 ...]
//
// Needs `npm run build`, and Google Chrome (CHROME_PATH, or the usual macOS / Linux places).
// The browser build is bundled with esbuild into a temporary directory, served from this
// machine and run in headless Chrome; ONNX Runtime's .wasm comes from node_modules. Each video
// is then run through the Node build, and the two are compared. Nothing is kept on disk.
import { build } from "esbuild";
import { createReadStream, existsSync, mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { basename, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { detect, findTransitions, labelFrames, loadModel } from "../dist/index.js";

const [backend, ...videos] = process.argv.slice(2);
if (!["webgpu", "wasm", "auto"].includes(backend) || !videos.length) {
  console.error("usage: node scripts/browser_parity.mjs webgpu|wasm|auto VIDEO.mp4 [VIDEO.mp4 ...]");
  process.exit(2);
}
const root = fileURLToPath(new URL("..", import.meta.url));
const chrome = [process.env.CHROME_PATH, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"].find(p => p && existsSync(p));
if (!chrome) throw new Error("Chrome not found; set CHROME_PATH");

const dir = mkdtempSync(join(tmpdir(), "leanscene-parity-"));
writeFileSync(join(dir, "app.js"), `import { detect, loadModel } from ${JSON.stringify(join(root, "dist/browser/index.js"))};
let model;
window.run = async (url, backend) => {
  model ??= await loadModel({ wasmPaths: "/ort/", ...(backend === "auto" ? {} : { backend }) });
  const t = performance.now(), r = await detect(url, { model });
  return { backend: model.backend, ms: performance.now() - t, frames: r.frames, fps: r.fps, probs: Array.from(r.probs) };
};
window.ready = true;\n`);
await build({ entryPoints: [join(dir, "app.js")], bundle: true, format: "esm", splitting: true, platform: "browser",
              outdir: join(dir, "out"), logLevel: "warning" });
const files = new Map(videos.map((v, i) => [`/v${i}${extname(v)}`, v]));
const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".mp4": "video/mp4", ".mov": "video/quicktime", ".wasm": "application/wasm" };
const server = createServer((req, res) => {
  const url = req.url.split("?")[0];
  if (url === "/") {
    return res.writeHead(200, { "content-type": "text/html" })
      .end('<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><script type="module" src="/out/app.js"></script>');
  }
  const path = files.get(url) ?? (url.startsWith("/ort/") ? join(root, "node_modules/onnxruntime-web/dist", url.slice(5)) : join(dir, url));
  if (!existsSync(path)) return res.writeHead(404).end();
  res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream", "content-length": statSync(path).size });
  createReadStream(path).pipe(res);
}).listen(0);
const browser = await puppeteer.launch({ executablePath: chrome, headless: "new", protocolTimeout: 3600000,
                                         args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--no-sandbox"] });
const page = await browser.newPage();
page.on("pageerror", e => console.error("[page]", String(e).slice(0, 300)));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.waitForFunction("window.ready === true", { timeout: 60000 });

// transitions of a with no transition of b within tol frames
const unmatched = (a, b, tol = 2) => a.filter(x => !b.some(y => x.start - tol <= y.end && y.start <= x.end + tol)).length;
const model = await loadModel();
const tot = { frames: 0, node: 0, web: 0, onlyNode: 0, onlyWeb: 0, same: 0, webMs: 0, nodeMs: 0, mean: [], max: 0, backend: "" };
for (const [url, v] of files) {
  let w;
  try {
    w = await page.evaluate((u, b) => window.run(u, b), url, backend);
  } catch (e) {
    console.log(`${basename(v)}: the browser failed: ${String(e).slice(0, 300)}`);
    continue;
  }
  const t = performance.now(), n = await detect(v, { model }), nodeMs = performance.now() - t;
  const wt = findTransitions(labelFrames(w.probs)), nt = n.transitions, m = Math.min(w.probs.length, n.probs.length);
  let sum = 0, max = 0;
  for (let i = 0; i < m; i++) {
    const d = Math.abs(w.probs[i] - n.probs[i]);
    sum += d;
    if (d > max) max = d;
  }
  tot.frames += n.frames; tot.node += nt.length; tot.web += wt.length; tot.onlyNode += unmatched(nt, wt); tot.onlyWeb += unmatched(wt, nt);
  tot.same += w.frames === n.frames; tot.webMs += w.ms; tot.nodeMs += nodeMs; tot.mean.push(sum / m); tot.max = Math.max(tot.max, max); tot.backend = w.backend;
  console.log(`${basename(v)}: frames ${n.frames} / ${w.frames}${w.frames === n.frames ? "" : "  DIFFERENT"}, fps ${n.fps?.toFixed(3)} / ${w.fps?.toFixed(3)}, ` +
    `transitions ${nt.length} / ${wt.length}, only Node ${unmatched(nt, wt)}, only browser ${unmatched(wt, nt)}, ` +
    `probability difference mean ${(sum / m).toFixed(4)} max ${max.toFixed(2)}, ${Math.round(n.frames / nodeMs * 1000)} / ${Math.round(w.frames / w.ms * 1000)} frames/s`);
}
console.log(`\n${files.size} videos, ${tot.frames.toLocaleString("en-US")} frames, browser on ${tot.backend}. Same frame count on ${tot.same} of ${files.size}.`);
console.log(`Node finds ${tot.node} transitions, the browser ${tot.web}; ${tot.onlyNode} only in Node, ${tot.onlyWeb} only in the browser (two-frame tolerance).`);
console.log(`Probability difference: mean ${(tot.mean.reduce((a, b) => a + b, 0) / tot.mean.length).toFixed(4)}, largest ${tot.max.toFixed(2)}.`);
console.log(`Speed end to end: Node ${Math.round(tot.frames / tot.nodeMs * 1000)} frames/s, browser ${Math.round(tot.frames / tot.webMs * 1000)} frames/s.`);
await browser.close();
server.close();
rmSync(dir, { recursive: true, force: true });
