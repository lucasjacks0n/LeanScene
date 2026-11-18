/** The decision rule and the steps from per-frame labels to transitions and shots. */

/** The rule used for every number in the paper. It was fixed before the test sets were scored. */
export const CUT_THRESHOLD = 0.7;
export const FADE_THRESHOLD = 0.8;
/** Fade frames separated by at most this many frames are joined into one run. */
export const FADE_CLOSE = 8;

/**
 * A maximal run of frames not labeled same. `start` and `end` are inclusive frame indices.
 *
 * `kind` is "cut" when the run holds no fade frame, else "gradual". For a cut, `end` is the
 * first frame of the new shot; start < end only when a few junk frames sit between two shots.
 */
export interface Transition {
  start: number;
  end: number;
  kind: "cut" | "gradual";
}

/** A shot as inclusive [first frame, last frame]. */
export type Shot = [number, number];

export interface RuleOptions {
  /** Cut threshold (default 0.7). Raise for fewer false alarms, lower to miss less. */
  cut?: number;
  /** Gradual-transition threshold (default 0.8). */
  fade?: number;
  /** Fade frames at most this many frames apart are joined (default 8). */
  close?: number;
}

/**
 * Apply the decision rule. `probs` holds three values per frame (same, cut, fade), row by row.
 * Returns one label per frame: 0 same, 1 cut, 2 fade.
 */
export function labelFrames(probs: ArrayLike<number>, options: RuleOptions = {}): Uint8Array {
  const { cut = CUT_THRESHOLD, fade = FADE_THRESHOLD, close = FADE_CLOSE } = options;
  const n = Math.floor(probs.length / 3);
  const labels = new Uint8Array(n);
  // Compared in single precision, as the Python package does: a float32 probability of 0.7 is
  // slightly below the double 0.7, and must still pass a threshold of 0.7.
  const c = Math.fround(cut), f = Math.fround(fade);
  for (let i = 0; i < n; i++) {
    if (Math.fround(probs[3 * i + 1]) >= c) labels[i] = 1;
    else if (Math.fround(probs[3 * i + 2]) >= f) labels[i] = 2;
  }
  let last = -1;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== 2) continue;
    if (last >= 0 && i - last > 1 && i - last <= close + 1) labels.fill(2, last, i);
    last = i;
  }
  return labels;
}

/** Runs of frames not labeled same, in order. */
export function findTransitions(labels: ArrayLike<number>): Transition[] {
  const out: Transition[] = [];
  const n = labels.length;
  let i = 0;
  while (i < n) {
    if (labels[i] === 0) {
      i++;
      continue;
    }
    const start = i;
    let gradual = false;
    while (i < n && labels[i] !== 0) {
      if (labels[i] === 2) gradual = true;
      i++;
    }
    out.push({ start, end: i - 1, kind: gradual ? "gradual" : "cut" });
  }
  return out;
}

/**
 * Shots as inclusive [first frame, last frame] pairs.
 *
 * A cut starts the new shot on its last frame. The frames of a gradual transition belong to
 * neither neighbor.
 */
export function findShots(transitions: Iterable<Transition>, frames: number): Shot[] {
  const shots: Shot[] = [];
  let start = 0;
  for (const t of transitions) {
    if (t.start - 1 >= start) shots.push([start, t.start - 1]);
    start = t.kind === "cut" ? t.end : t.end + 1;
  }
  if (start <= frames - 1) shots.push([start, frames - 1]);
  return shots;
}
