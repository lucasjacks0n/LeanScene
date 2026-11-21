// Shot search for the LeanScene demo (index.html?clip, or search.html).
// One frame from the middle of every shot is embedded with a CLIP image encoder; a typed
// description is embedded with the matching text encoder, and the shots are ranked by cosine
// similarity. The frames are picked up while the video is read the first time (see "frames kept
// from the first pass" below), so embedding does not have to decode the video again; it starts
// when the cuts are on screen, so that finding them is not slowed down. The models run through
// transformers.js in a worker of this tab and are downloaded from the Hugging Face hub on first
// use (the browser caches them).
//
// A plain script, not a module file: a browser refuses to load a module file from disk
// (file://), and the demo is meant to work when index.html is opened directly. The worker is
// started from a function in this file for the same reason.
(async () => {
const LIB = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODELS = {
  // MobileCLIP-S0 (Apple): a CLIP-style model built for speed. 45 MB image encoder + 43 MB text encoder.
  // Its half-precision image encoder does not load on WebGPU (onnxruntime-web 1.22), so full precision is used.
  mobileclip: { id: "Xenova/mobileclip_s0", label: "MobileCLIP-S0", gpu: "fp32" },
  // OpenAI CLIP ViT-B/32, the original: 176 MB image encoder + 64 MB text encoder. ?clip=openai
  openai: { id: "Xenova/clip-vit-base-patch32", label: "CLIP ViT-B/32", gpu: "fp16" },
};
const M = MODELS[new URLSearchParams(location.search).get("clip")] || MODELS.mobileclip;
const BATCH = 8, TOP = 48;  // TOP: as many results as one page of transition cards
const el = id => document.getElementById(id);

const style = document.createElement("style");
style.textContent = `
  #search { margin:26px 0 2px; }
  #search h2 { margin-top:0; }
  #qrow { display:flex; gap:8px; }
  #q { flex:1; min-width:0; font:inherit; font-size:16px; /* 16px or more, or iOS Safari zooms the page when the field is focused */ padding:9px 12px; border-radius:8px; border:1px solid var(--line); background:var(--panel); color:var(--ink); }
  #q:focus { outline:none; border-color:var(--accent); }
  #cstat { font-size:13px; margin:8px 0 12px; }
  .hit { position:relative; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; cursor:pointer; padding:0; text-align:left; color:inherit; font:inherit; }
  .hit:hover, .hit:focus-visible { border-color:var(--accent); outline:none; }
  .hit img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; background:#111; font-size:0; }
  .hit img.loading { background:linear-gradient(100deg, var(--line) 30%, var(--panel) 50%, var(--line) 70%) 0 0 / 200% 100%; animation:shimmer 1.2s linear infinite; }
  .hit .cap { padding:6px 9px 8px; font-size:12px; }
  .hit .like { position:absolute; top:6px; right:6px; font-size:11px; padding:2px 7px; border-radius:6px; border:0; background:rgba(0,0,0,.6); color:#fff; cursor:pointer; opacity:0; }
  .hit:hover .like, .hit:focus-within .like { opacity:1; }
`;
document.head.append(style);

const section = document.createElement("section");
section.id = "search";
section.innerHTML = `
  <h2>Search the shots</h2>
  <div id="qrow"><input id="q" type="text" placeholder="Describe a shot: a city at night, two people talking, a close-up of a face…" autocomplete="off" disabled>
    <button class="plain" id="qclear" hidden>Clear</button></div>
  <p id="cstat" class="mute"></p>
`;
// Results are shown where the transition cards are, so the search sits right above that list.
const cardsRow = el("cardsTitle").closest(".row"), cardsHelp = cardsRow.nextElementSibling;
cardsRow.parentNode.insertBefore(section, cardsRow);
// While a search is showing, the page's own card rendering stands aside (its paging and the
// highlight that follows the arrow keys would otherwise overwrite the results).
const pageShow = showPage, pageMark = markCard;
const searching = () => !!(state && state.result === result && state.query);
showPage = (p, scroll) => { if (!searching()) pageShow(p, scroll); };
markCard = k => { if (searching()) activeCard = k; else pageMark(k); };
function restoreCards(S) {  // back to the transition cards
  if (!S || S.result !== result) return;
  const n = result.transitions.length;
  el("cardsTitle").textContent = n ? `${n} transitions` : "No transitions found";
  if (cardsHelp && cardsHelp.tagName === "P") cardsHelp.hidden = false;
  pageShow(cardPage);
}

let models = null, state = null;  // state: the shots and embeddings of the video on screen
window.clipState = () => state && { shots: state.shots.length, done: state.done, kept: state.kept, dim: state.dim, model: M.id, device: models && models.device, fallback: models && models.fallback, seconds: state.seconds, error: state.error };

function shotsOf(R) {
  const out = []; let start = 0;
  for (const t of R.transitions) { if (t.before >= start) out.push({ start, end: t.before }); start = t.after; }
  if (start <= R.frames - 1) out.push({ start, end: R.frames - 1 });
  for (const s of out) { s.mid = (s.start + s.end) >> 1; s.rep = s.mid; }  // rep: the frame shown and embedded for the shot
  return out;
}

// Loading state, for pages that show it (window.clipLoad): phase idle | loading | ready | error,
// megabytes downloaded so far and in total, and the device the image encoder runs on.
const load = window.clipLoad = { phase: "idle", loaded: 0, total: 0, device: null, label: M.label, error: null };
let loading = null;
const reporters = new Set();
function loadModels(report) {  // one download, however many callers
  if (report) reporters.add(report);
  if (!loading) loading = loadOnce(msg => reporters.forEach(r => r(msg))).catch(e => { load.phase = "error"; load.error = String(e && e.message || e); loading = null; throw e; });
  return loading;
}
window.clipPreload = () => loadModels().catch(e => console.error(e));
async function loadOnce(report) {
  load.phase = "loading"; load.error = null;
  onProgress = m => {
    load.loaded = m.loaded / 1e6; load.total = m.total / 1e6;
    report(`Downloading ${M.label}: ${load.loaded.toFixed(0)} of ${load.total.toFixed(0)} MB (first visit only)`);
  };
  report(`Loading ${M.label}…`);
  const r = await ask("load", { lib: LIB, id: M.id, gpu: M.gpu });
  if (r.fallback) console.warn("CLIP image encoder: fell back to WebAssembly:", r.fallback);
  load.phase = "ready"; load.device = r.device;
  return (models = r);
}

// --- the worker --------------------------------------------------------------------------------
// Everything that touches the models. This function is turned into text to start the worker, so
// it can use nothing from outside itself. It does one job at a time, in the order they were sent:
//   load   { lib, id, gpu }              -> { device, fallback }
//   images [{ kept: key } | URL | null]  -> [unit vector | null]   (null: no picture, or a flat one)
//   text   "a description"               -> unit vector
// and, at once, without waiting for the jobs above (they need no model):
//   keep   { key, bitmap }               -> true when the picture was kept (false: flat or unreadable)
//   forget                               -> drops every kept picture
function clipWorker() {
  let T, models = null, chain = Promise.resolve();
  const kept = new Map();  // key -> the picture as a JPEG (a few kilobytes each, so a whole film's shots fit)
  async function keep({ key, bitmap }) {
    try {
      const c = new OffscreenCanvas(bitmap.width, bitmap.height), x = c.getContext("2d", { willReadFrequently: true });
      x.drawImage(bitmap, 0, 0); bitmap.close();
      if (contrast({ data: x.getImageData(0, 0, c.width, c.height).data, width: c.width, height: c.height, channels: 4 }) < FLAT) return false;
      kept.set(key, await c.convertToBlob({ type: "image/jpeg", quality: 0.9 }));
      return true;
    } catch (_) { return false; }
  }
  const unit = v => { let n = 0; for (let i = 0; i < v.length; i++) n += v[i] * v[i]; n = Math.sqrt(n) || 1; for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
  // A frame with almost no contrast (black, or one flat color) says nothing about its shot.
  // Measured as the spread of brightness over a sample of pixels, on a 0-255 scale.
  const FLAT = 6;
  function contrast(img) {
    const d = img.data, c = img.channels, n = img.width * img.height, step = Math.max(1, Math.floor(n / 4000));
    let sum = 0, sq = 0, m = 0;
    for (let p = 0; p < n; p += step) { const o = p * c, y = c >= 3 ? 0.299 * d[o] + 0.587 * d[o + 1] + 0.114 * d[o + 2] : d[o]; sum += y; sq += y * y; m++; }
    const mean = sum / m; return Math.sqrt(Math.max(0, sq / m - mean * mean));
  }
  async function load({ lib, id, gpu }) {
    T = await import(lib);
    const seen = {};
    const progress_callback = p => {
      if (p.status !== "progress" || !/\.onnx$/.test(p.file || "")) return;
      seen[p.file] = [p.loaded, p.total];
      const [loaded, total] = Object.values(seen).reduce((s, v) => [s[0] + v[0], s[1] + v[1]], [0, 0]);
      postMessage({ type: "progress", loaded, total });
    };
    let device = "wasm", dtype = "q8";
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter().catch(() => null);
      if (adapter) { device = "webgpu"; dtype = adapter.features.has("shader-f16") ? gpu : "fp32"; }
    }
    const [processor, tokenizer, text] = await Promise.all([
      T.AutoProcessor.from_pretrained(id), T.AutoTokenizer.from_pretrained(id),
      T.CLIPTextModelWithProjection.from_pretrained(id, { device: "wasm", dtype: "q8", progress_callback }),
    ]);
    let vision, fallback = null;
    try { vision = await T.CLIPVisionModelWithProjection.from_pretrained(id, { device, dtype, progress_callback }); }
    catch (e) {
      if (device === "wasm") throw e;
      fallback = String(e && e.message || e); device = "wasm"; vision = await T.CLIPVisionModelWithProjection.from_pretrained(id, { device, dtype: "q8", progress_callback });
    }
    models = { processor, tokenizer, text, vision };
    return { device, fallback };
  }
  async function picture(src) {  // null when it cannot be read, or is flat
    try {
      const blob = typeof src === "string" ? null : kept.get(src.kept);
      if (!blob && typeof src !== "string") return null;
      const img = blob ? await T.RawImage.fromBlob(blob) : await T.RawImage.fromURL(src);
      return contrast(img) >= FLAT ? img : null;
    } catch (_) { return null; }
  }
  async function images(srcs) {
    const imgs = await Promise.all(srcs.map(s => s ? picture(s) : null)), ok = imgs.filter(Boolean), out = imgs.map(() => null);
    if (!ok.length) return out;
    const { image_embeds } = await models.vision(await models.processor(ok));
    const d = image_embeds.dims[1]; let k = 0;
    imgs.forEach((img, i) => { if (img) { out[i] = unit(image_embeds.data.slice(k * d, (k + 1) * d)); k++; } });
    return out;
  }
  async function text(t) {
    const { text_embeds } = await models.text(models.tokenizer([t], { padding: "max_length", truncation: true }));
    return unit(Float32Array.from(text_embeds.data));
  }
  const jobs = { load, images, text };
  self.onmessage = ev => {
    const { id, type, arg } = ev.data;
    if (type === "keep") return keep(arg).then(result => postMessage({ id, result }));
    if (type === "forget") { kept.clear(); return postMessage({ id, result: true }); }
    chain = chain.then(() => jobs[type](arg)).then(result => postMessage({ id, result }), e => postMessage({ id, error: String(e && e.message || e) }));
  };
}

let worker = null, asked = 0, onProgress = () => {};
const answers = new Map();
function ask(type, arg, transfer) {
  if (!worker) {
    worker = new Worker(URL.createObjectURL(new Blob(["(" + clipWorker + ")()"], { type: "text/javascript" })));
    worker.onmessage = ev => {
      const m = ev.data;
      if (m.type === "progress") return onProgress(m);
      const a = answers.get(m.id); if (!a) return;
      answers.delete(m.id); m.error ? a.no(new Error(m.error)) : a.yes(m.result);
    };
    worker.onerror = ev => { for (const a of answers.values()) a.no(new Error(ev.message || "the search worker stopped")); answers.clear(); };
  }
  return new Promise((yes, no) => { const id = ++asked; answers.set(id, { yes, no }); worker.postMessage({ id, type, arg }, transfer || []); });
}
const embedImages = srcs => ask("images", srcs);

// One vector per shot, null for a shot left out of the search. The picture that stands for a
// shot is the frame kept for it from the first pass when there is one (within a few frames of
// its middle), else its middle frame, decoded from the file; when that is flat (a fade caught
// at its darkest), the first of a few others that is not. A shot where all of them are flat is
// left out.
async function embedBatch(R, shots) {
  const K = keeping && keeping.src.thumbs === R.thumbs ? keeping : null;
  const first = await Promise.all(shots.map(async s => {
    const frame = K && K.has.get(keyOf(s));
    if (frame !== undefined) return { src: { kept: keyOf(s) }, frame, kept: true };
    const url = await R.thumbs.get(s.mid);
    return url ? { src: url, frame: s.mid } : null;
  }));
  const vecs = await embedImages(first.map(x => x && x.src));
  await Promise.all(shots.map(async (s, k) => {
    if (vecs[k]) { s.rep = first[k].frame; if (first[k].kept) R.keptUsed = (R.keptUsed || 0) + 1; return; }
    const len = s.end - s.start, tried = new Set([first[k] ? first[k].frame : s.mid]);
    for (const frame of [s.mid, ...[0.25, 0.75, 0.1, 0.9].map(f => s.start + Math.round(len * f))]) {
      if (tried.has(frame)) continue;
      tried.add(frame);
      const url = await R.thumbs.get(frame);
      const [v] = url ? await embedImages([url]) : [null];
      if (v) { vecs[k] = v; s.rep = frame; return; }
    }
    s.blank = true;
  }));
  return vecs;
}
function put(S, j, vec) {
  if (!vec) return;
  if (!S.emb) { S.dim = vec.length; S.emb = new Float32Array(S.shots.length * S.dim); }
  S.emb.set(vec, j * S.dim); S.have[j] = 1;
}

async function embedShots(S) {
  const R = S.result, t0 = performance.now();
  for (let i = 0; i < S.shots.length; i += BATCH) {
    const idx = []; for (let j = i; j < Math.min(S.shots.length, i + BATCH); j++) idx.push(j);
    const vecs = await embedBatch(R, idx.map(j => S.shots[j]));
    if (state !== S) return;  // another video was dropped
    idx.forEach((j, k) => put(S, j, vecs[k]));
    S.done = Math.min(S.shots.length, i + BATCH); S.seconds = (performance.now() - t0) / 1000; S.kept = R.keptUsed || 0;
    status(S);
    if (S.query) rank(S);  // results fill in as more shots are embedded
  }
}

// --- frames kept from the first pass -----------------------------------------------------------
// While it reads the video, the page hands on each shot as soon as its end is known
// (window.LEANSCENE_STREAM, and a "leanscene:shots" event each time it grows). A shot may come
// with a small bitmap of a frame near its middle, held from that first pass (`held`, frame
// number `heldFrame`). It goes straight to the worker, which keeps it as a JPEG until the shots
// are embedded: that way most shots need no second decode of the video. Nothing is embedded
// here, so that reading the video is not slowed down.
let keeping = null;  // { src, has: shot -> the frame kept for it, pending: answers still awaited }
const keyOf = s => s.start + "-" + s.end;
window.clipHoldFrames = () => true;  // asks the page to hold frames for the shots it hands on
document.addEventListener("leanscene:shots", () => {
  const src = window.LEANSCENE_STREAM;
  if (!src) return;
  if (!keeping || keeping.src !== src) { keeping = { src, has: new Map(), pending: [], next: 0 }; ask("forget"); }
  const K = keeping;
  for (; K.next < src.shots.length; K.next++) {
    const s = src.shots[K.next]; if (!s.held) continue;
    const p = s.held, key = keyOf(s), frame = s.heldFrame; s.held = null;
    K.pending.push(p.then(b => ask("keep", { key, bitmap: b }, [b])).then(ok => { if (ok) K.has.set(key, frame); }, () => {}));
  }
});

function status(S) {
  const n = S.shots.length, where = models.device === "webgpu" ? "on your GPU" : "on your CPU";
  el("cstat").textContent = S.done < n
    ? `Embedding shots with ${M.label} ${where}: ${S.done.toLocaleString()} of ${n.toLocaleString()}. You can search already.`
    : `${n.toLocaleString()} shots embedded with ${M.label} ${where} in ${S.seconds.toFixed(1)} s, one frame per shot.${S.shots.some(x => x.blank) ? ` ${S.shots.filter(x => x.blank).length} blank shots (all black or one flat color) are left out.` : ""}`;
}

function rank(S) {
  if (!S.emb || !S.query) return;
  const d = S.dim, q = S.query.vec, scored = [];
  for (let j = 0; j < S.shots.length; j++) {
    if (!S.have[j] || j === S.query.self) continue;
    let dot = 0; for (let i = 0; i < d; i++) dot += q[i] * S.emb[j * d + i];
    scored.push([dot, j]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  const top = scored.slice(0, TOP), key = top.map(x => x[1]).join(",");
  if (key === S.shown) return;
  S.shown = key;
  el("cardsTitle").textContent = S.query.label(top.length);
  if (cardsHelp && cardsHelp.tagName === "P") cardsHelp.hidden = true;
  document.querySelectorAll(".pager").forEach(e => e.hidden = true);
  el("cards").replaceChildren(...top.map(([score, j]) => {
    const s = S.shots[j], b = document.createElement("div");
    b.className = "hit"; b.tabIndex = 0; b.setAttribute("role", "button");
    b.innerHTML = `<img alt="shot ${j + 1}"><button class="like" title="Find shots that look like this one">Similar</button>` +
      `<div class="cap"><span>${fmtTime(timeOf(s.start))}</span> <span class="mute">· shot ${j + 1} · ${score.toFixed(2)}</span></div>`;
    S.result.thumbs.lazy(b.querySelector("img"), s.rep);
    const play = () => { seek(timeOf(s.start) + 0.001, true); el("player").scrollIntoView({ behavior: "smooth", block: "nearest" }); };
    b.onclick = play; b.onkeydown = ev => { if (ev.key === "Enter") play(); };
    b.querySelector(".like").onclick = ev => { ev.stopPropagation(); similar(S, j); };
    return b;
  }));
}

async function search(S, text) {
  text = text.trim();
  el("qclear").hidden = !text;
  if (!text) { const was = S.query; S.query = null; S.shown = ""; if (was) restoreCards(S); return; }
  const ticket = ++S.ticket;
  const vec = await ask("text", text);
  if (state !== S || ticket !== S.ticket) return;  // a newer query overtook this one
  S.query = { vec, self: -1, label: n => `Top ${n} shots for “${text}”` }; S.shown = ""; rank(S);
}

function similar(S, j) {
  S.ticket++;
  el("q").value = ""; el("qclear").hidden = false;
  S.query = { vec: S.emb.slice(j * S.dim, (j + 1) * S.dim), self: j, label: n => `${n} shots that look like shot ${j + 1}` }; S.shown = ""; rank(S);
  el("cstat").textContent = `Shots that look like shot ${j + 1} (${fmtTime(timeOf(S.shots[j].start))}).`;
}

let typing = 0;
el("q").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(() => state && state.ready && search(state, el("q").value), 180); });
el("q").addEventListener("keydown", ev => ev.stopPropagation());  // arrow keys edit the text here, not step transitions
el("qclear").onclick = () => { el("q").value = ""; if (state) { search(state, ""); if (state.ready) status(state); } el("q").focus(); };

async function start() {
  const R = result;
  const S = state = { result: R, shots: shotsOf(R), have: null, kept: 0, emb: null, dim: 0, done: 0, seconds: 0, query: null, shown: "", ticket: 0, ready: false, error: null };
  S.have = new Uint8Array(S.shots.length);
  el("q").value = ""; el("q").disabled = true; el("qclear").hidden = true;
  if (cardsHelp && cardsHelp.tagName === "P") cardsHelp.hidden = false;
  try {
    await loadModels(msg => { if (state === S) el("cstat").textContent = msg; });
    if (state !== S) return;
    if (keeping && keeping.src.thumbs === R.thumbs) await Promise.all(keeping.pending);  // the last frames handed to the worker
    if (state !== S) return;
    S.ready = true; el("q").disabled = false; status(S);
    await embedShots(S);
  } catch (e) {
    console.error(e); S.error = String(e && e.message || e);
    if (state === S) el("cstat").textContent = "Shot search could not start: " + S.error;
  }
}
document.addEventListener("leanscene:result", start);
if (result) start();  // a video finished before this script loaded
// A page that sets window.CLIP_PRELOAD starts the download right away (not when the browser asks to save data).
else if (window.CLIP_PRELOAD && !(navigator.connection && navigator.connection.saveData)) window.clipPreload();
})();
