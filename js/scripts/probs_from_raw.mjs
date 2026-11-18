// Used by parity.py: run the model over frames already decoded and resized by OpenCV.
//   node probs_from_raw.mjs FRAMES.raw OUT.npy     (FRAMES.raw: 100x100 BGR frames, back to back)
import { readFileSync, writeFileSync } from "node:fs";
import { FRAME_BYTES, frameProbs } from "../dist/index.js";

const [raw, out] = process.argv.slice(2);
const bytes = readFileSync(raw);
const frames = [];
for (let at = 0; at + FRAME_BYTES <= bytes.length; at += FRAME_BYTES) frames.push(bytes.subarray(at, at + FRAME_BYTES));
const probs = await frameProbs(frames);
let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${probs.length / 3}, 3), }`;
header += " ".repeat(63 - ((10 + header.length) % 64)) + "\n";
writeFileSync(out, Buffer.concat([Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]),
                                  Buffer.from(header), Buffer.from(probs.buffer, probs.byteOffset, probs.byteLength)]));
