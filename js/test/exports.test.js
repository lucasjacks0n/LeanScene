// Which build `import ... from "leanscene"` gets: Node gets the Node build, a bundler for the
// browser gets the browser build, and TypeScript gets the matching types with no settings
// (moduleResolution "bundler" -> browser, "nodenext" -> Node). The package is linked into a
// temporary project under its own name, as an install would place it.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
let dir;
before(() => {
  dir = mkdtempSync(join(tmpdir(), "leanscene-exports-"));
  mkdirSync(join(dir, "node_modules"));
  symlinkSync(root, join(dir, "node_modules", "leanscene"), "dir");
  writeFileSync(join(dir, "package.json"), JSON.stringify({ type: "module" }));
});
after(() => { if (dir) rmSync(dir, { recursive: true, force: true }); });

test("Node gets the Node build", () => {
  writeFileSync(join(dir, "node.js"), 'import * as l from "leanscene";\nconsole.log(JSON.stringify(["ffmpegPath" in l, "frameFrom" in l]));\n');
  assert.equal(execFileSync(process.execPath, [join(dir, "node.js")], { encoding: "utf8" }).trim(), "[true,false]");
});

test("a bundler for the browser gets the browser build", async () => {
  const { build } = await import("esbuild");
  writeFileSync(join(dir, "web.js"), 'import { detect, frameFrom } from "leanscene";\nconsole.log(detect, frameFrom);\n');
  const r = await build({ entryPoints: [join(dir, "web.js")], bundle: true, format: "esm", platform: "browser", write: false,
                          metafile: true, external: ["onnxruntime-web", "onnxruntime-web/*", "mp4box"], absWorkingDir: dir });
  const inputs = Object.keys(r.metafile.inputs);
  assert.ok(inputs.some(f => f.includes("dist/browser/index.js")), inputs.join("\n"));
  assert.ok(!inputs.some(f => /dist\/(index|model|decode)\.js$/.test(f)), "the Node build is in the browser bundle");
});

test("TypeScript gets the browser types for a bundler and the Node types for nodenext, with no settings", () => {
  const tsc = join(root, "node_modules", "typescript", "bin", "tsc");
  const check = (file, code, ...flags) => {
    writeFileSync(join(dir, file), code);
    try {
      execFileSync(process.execPath, [tsc, "--noEmit", "--strict", "--skipLibCheck", "--target", "es2022", "--lib", "es2022,dom",
                                      ...flags, join(dir, file)], { encoding: "utf8" });
      return "ok";
    } catch (e) {
      return e.stdout;
    }
  };
  const web = 'import { detect, loadModel } from "leanscene";\n'
    + 'export async function f(file: File) { await loadModel({ wasmPaths: "/ort/" }); return detect(file); }\n';
  assert.equal(check("web.ts", web, "--module", "esnext", "--moduleResolution", "bundler"), "ok");
  const node = 'import { detect, ffmpegPath } from "leanscene";\n'
    + 'export async function f(path: string) { ffmpegPath(); return detect(path); }\n';
  assert.equal(check("node.ts", node, "--module", "nodenext", "--moduleResolution", "nodenext", "--types", "node",
                     "--typeRoots", join(root, "node_modules", "@types")), "ok");
});
