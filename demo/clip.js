// Shot search for the LeanScene demo (index.html?clip, or search.html).
// When a video has been processed, one frame from the middle of every shot is embedded with a
// CLIP image encoder; a typed description is embedded with the matching text encoder, and the
// shots are ranked by cosine similarity. Models run in this tab through transformers.js and are
// downloaded from the Hugging Face hub on first use (the browser caches them).
//
// A plain script with a dynamic import, not a module file: a browser refuses to load a module
// file from disk (file://), and the demo is meant to work when index.html is opened directly.
(async () => {
const note = document.getElementById("searchnote");
let lib;
try { lib = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1"); }
catch (e) { console.error(e); note.textContent = "Shot search could not load (it needs an internet connection): " + (e && e.message || e); return; }
const { AutoProcessor, AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage } = lib;

const MODELS = {
  // MobileCLIP-S0 (Apple): a CLIP-style model built for speed. 45 MB image encoder + 43 MB text encoder.
  // Its half-precision image encoder does not load on WebGPU (onnxruntime-web 1.22), so full precision is used.
  mobileclip: { id: "Xenova/mobileclip_s0", label: "MobileCLIP-S0", gpu: "fp32" },
  // OpenAI CLIP ViT-B/32, the original: 176 MB image encoder + 64 MB text encoder. ?clip=openai
  openai: { id: "Xenova/clip-vit-base-patch32", label: "CLIP ViT-B/32", gpu: "fp16" },
};
const M = MODELS[new URLSearchParams(location.search).get("clip")] || MODELS.mobileclip;
const BATCH = 8, TOP = 24;
const el = id => document.getElementById(id);

const style = document.createElement("style");
style.textContent = `
  #search { margin:0 0 18px; }
  #search h2 { margin-top:0; }
  #qrow { display:flex; gap:8px; }
  #q { flex:1; min-width:0; font:inherit; font-size:15px; padding:9px 12px; border-radius:8px; border:1px solid var(--line); background:var(--panel); color:var(--ink); }
  #q:focus { outline:none; border-color:var(--accent); }
  #cstat { font-size:13px; margin:8px 0 12px; }
  #hits { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:10px; }
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
  <div id="hits"></div>`;
el("summary").after(section);  // first thing in the results, above the player

let models = null, state = null;  // state: the shots and embeddings of the video on screen
window.clipState = () => state && { shots: state.shots.length, done: state.done, dim: state.dim, model: M.id, device: models && models.device, fallback: models && models.fallback, seconds: state.seconds, error: state.error };

function shotsOf(R) {
  const out = []; let start = 0;
  for (const t of R.transitions) { if (t.before >= start) out.push({ start, end: t.before }); start = t.after; }
  if (start <= R.frames - 1) out.push({ start, end: R.frames - 1 });
  for (const s of out) s.mid = (s.start + s.end) >> 1;
  return out;
}

async function loadModels(report) {
  if (models) return models;
  const seen = {};
  const progress_callback = p => {
    if (p.status !== "progress" || !/\.onnx$/.test(p.file || "")) return;
    seen[p.file] = [p.loaded, p.total];
    const [a, b] = Object.values(seen).reduce((s, v) => [s[0] + v[0], s[1] + v[1]], [0, 0]);
    report(`Downloading ${M.label}: ${(a / 1e6).toFixed(0)} of ${(b / 1e6).toFixed(0)} MB (first visit only)`);
  };
  let device = "wasm", dtype = "q8";
  if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (adapter) { device = "webgpu"; dtype = adapter.features.has("shader-f16") ? M.gpu : "fp32"; }
  }
  report(`Loading ${M.label}…`);
  const [processor, tokenizer, text] = await Promise.all([
    AutoProcessor.from_pretrained(M.id), AutoTokenizer.from_pretrained(M.id),
    CLIPTextModelWithProjection.from_pretrained(M.id, { device: "wasm", dtype: "q8", progress_callback }),
  ]);
  let vision, fallback = null;
  try { vision = await CLIPVisionModelWithProjection.from_pretrained(M.id, { device, dtype, progress_callback }); }
  catch (e) {
    if (device === "wasm") throw e;
    console.warn("CLIP image encoder: falling back to WebAssembly:", e);
    fallback = String(e && e.message || e); device = "wasm"; vision = await CLIPVisionModelWithProjection.from_pretrained(M.id, { device, dtype: "q8", progress_callback });
  }
  return (models = { processor, tokenizer, text, vision, device, fallback });
}

const unit = (v, o, d) => { let n = 0; for (let i = 0; i < d; i++) n += v[o + i] * v[o + i]; n = Math.sqrt(n) || 1; for (let i = 0; i < d; i++) v[o + i] /= n; };

async function embedShots(S) {
  const R = S.result, t0 = performance.now();
  for (let i = 0; i < S.shots.length; i += BATCH) {
    const idx = []; for (let j = i; j < Math.min(S.shots.length, i + BATCH); j++) idx.push(j);
    const urls = await Promise.all(idx.map(j => R.thumbs.get(S.shots[j].mid)));
    if (state !== S) return;  // another video was dropped
    const ok = idx.filter((_, k) => urls[k]);
    if (ok.length) {
      const images = await Promise.all(urls.filter(Boolean).map(u => RawImage.fromURL(u)));
      const { image_embeds } = await models.vision(await models.processor(images));
      const d = image_embeds.dims[1];
      if (!S.emb) { S.dim = d; S.emb = new Float32Array(S.shots.length * d); }
      ok.forEach((j, k) => { S.emb.set(image_embeds.data.subarray(k * d, (k + 1) * d), j * d); unit(S.emb, j * d, d); S.have[j] = 1; });
    }
    S.done = Math.min(S.shots.length, i + BATCH); S.seconds = (performance.now() - t0) / 1000;
    status(S);
    if (S.query) rank(S);  // results fill in as more shots are embedded
  }
}

function status(S) {
  const n = S.shots.length, where = models.device === "webgpu" ? "on your GPU" : "on your CPU";
  el("cstat").textContent = S.done < n
    ? `Embedding shots with ${M.label} ${where}: ${S.done.toLocaleString()} of ${n.toLocaleString()}. You can search already.`
    : `${n.toLocaleString()} shots embedded with ${M.label} ${where} in ${S.seconds.toFixed(1)} s. One frame from the middle of each shot.`;
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
  el("hits").replaceChildren(...top.map(([score, j]) => {
    const s = S.shots[j], b = document.createElement("div");
    b.className = "hit"; b.tabIndex = 0; b.setAttribute("role", "button");
    b.innerHTML = `<img alt="shot ${j + 1}"><button class="like" title="Find shots that look like this one">Similar</button>` +
      `<div class="cap"><span>${fmtTime(timeOf(s.start))}</span> <span class="mute">· shot ${j + 1} · ${score.toFixed(2)}</span></div>`;
    S.result.thumbs.lazy(b.querySelector("img"), s.mid);
    const play = () => { seek(timeOf(s.start) + 0.001, true); el("player").scrollIntoView({ behavior: "smooth", block: "nearest" }); };
    b.onclick = play; b.onkeydown = ev => { if (ev.key === "Enter") play(); };
    b.querySelector(".like").onclick = ev => { ev.stopPropagation(); similar(S, j); };
    return b;
  }));
}

async function search(S, text) {
  text = text.trim();
  el("qclear").hidden = !text;
  if (!text) { S.query = null; S.shown = ""; el("hits").replaceChildren(); return; }
  const ticket = ++S.ticket;
  const { text_embeds } = await models.text(models.tokenizer([text], { padding: "max_length", truncation: true }));
  if (state !== S || ticket !== S.ticket) return;  // a newer query overtook this one
  const vec = Float32Array.from(text_embeds.data); unit(vec, 0, vec.length);
  S.query = { vec, self: -1 }; S.shown = ""; rank(S);
}

function similar(S, j) {
  S.ticket++;
  el("q").value = ""; el("qclear").hidden = false;
  S.query = { vec: S.emb.slice(j * S.dim, (j + 1) * S.dim), self: j }; S.shown = ""; rank(S);
  el("cstat").textContent = `Shots that look like shot ${j + 1} (${fmtTime(timeOf(S.shots[j].start))}).`;
}

let typing = 0;
el("q").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(() => state && state.ready && search(state, el("q").value), 180); });
el("q").addEventListener("keydown", ev => ev.stopPropagation());  // arrow keys edit the text here, not step transitions
el("qclear").onclick = () => { el("q").value = ""; if (state) { search(state, ""); if (state.ready) status(state); } el("q").focus(); };

async function start() {
  const R = result;
  const S = state = { result: R, shots: shotsOf(R), have: null, emb: null, dim: 0, done: 0, seconds: 0, query: null, shown: "", ticket: 0, ready: false, error: null };
  S.have = new Uint8Array(S.shots.length);
  el("hits").replaceChildren(); el("q").value = ""; el("q").disabled = true; el("qclear").hidden = true;
  try {
    await loadModels(msg => { if (state === S) el("cstat").textContent = msg; });
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
})();
