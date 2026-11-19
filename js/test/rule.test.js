// The decision rule and the steps after it: the same cases as tests/test_rule.py.
import assert from "node:assert/strict";
import { test } from "node:test";
import { findShots, findTransitions, labelFrames } from "../dist/index.js";

const probs = labels => Float32Array.from(labels.flatMap(l => [0, 1, 2].map(k => +(k === l))));
const cut = (start, end = start) => ({ start, end, kind: "cut" });
const gradual = (start, end) => ({ start, end, kind: "gradual" });

test("thresholds", () => {
  const p = new Float32Array([0.31, 0.69, 0.0, 0.3, 0.7, 0.0, 0.21, 0.0, 0.79, 0.2, 0.0, 0.8]);
  assert.deepEqual([...labelFrames(p)], [0, 1, 0, 2]);
  assert.deepEqual([...labelFrames(p, { cut: 0.6, fade: 0.7 })], [1, 1, 2, 2]);
});

test("fade runs are joined over short gaps", () => {
  const gap8 = [2, ...Array(8).fill(0), 2], gap9 = [2, ...Array(9).fill(0), 2];
  assert.deepEqual([...labelFrames(probs(gap8))], Array(10).fill(2));
  assert.deepEqual([...labelFrames(probs(gap9))], gap9);
});

test("transitions and shots", () => {
  //              0  1  2  3  4  5  6  7  8  9 10 11
  const labels = [0, 0, 0, 1, 0, 0, 2, 2, 2, 0, 0, 0];
  const t = findTransitions(labels);
  assert.deepEqual(t, [cut(3), gradual(6, 8)]);
  // the cut frame opens the new shot; gradual frames belong to neither shot
  assert.deepEqual(findShots(t, labels.length), [[0, 2], [3, 5], [9, 11]]);
});

test("no transitions is one shot", () => {
  assert.deepEqual(findTransitions(new Uint8Array(5)), []);
  assert.deepEqual(findShots([], 5), [[0, 4]]);
  assert.deepEqual(findShots([], 0), []);
});

test("transitions at the edges", () => {
  const t = findTransitions([1, 0, 0, 2, 2]);
  assert.deepEqual(t, [cut(0), gradual(3, 4)]);
  assert.deepEqual(findShots(t, 5), [[0, 2]]);
});

test("junk frames between two shots", () => {
  // two cut labels in a row: the frame between the shots belongs to neither
  const t = findTransitions([0, 0, 1, 1, 0, 0]);
  assert.deepEqual(t, [cut(2, 3)]);
  assert.deepEqual(findShots(t, 6), [[0, 1], [3, 5]]);
});
