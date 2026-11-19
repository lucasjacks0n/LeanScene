"""Compare the npm package with the Python package on real videos.

    python js/scripts/parity.py VIDEO [VIDEO ...]

Needs both built: `pip install -e .` and `(cd js && npm install && npm run build)`. For each video:

  model   the same OpenCV-decoded frames through both packages (the first 3,000): what the port
          of the network itself changes
  whole   each package end to end on the file, with its own decoder and resize: what a user sees

and it times both end to end (model loading included, one run each). Nothing is written outside
a temporary directory, which is removed at the end.
"""
import json, os, subprocess, sys, tempfile, time

import cv2
import numpy as np

import leanscene

JS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(JS, "dist", "cli.js")
RAW = os.path.join(JS, "scripts", "probs_from_raw.mjs")
N = 3000


def events(p):
    return [(t.start, t.end) for t in leanscene.find_transitions(leanscene.label_frames(p))]


def unmatched(a, b, tol=2):
    """Transitions of a with no transition of b within tol frames."""
    return sum(1 for s, e in a if not any(s - tol <= e2 and s2 <= e + tol for s2, e2 in b))


model = leanscene.load_model(device="cpu")
tot = dict(frames=0, n_py=0, n_js=0, only_py=0, only_js=0, dp=[], dpmax=0.0, dm=0.0, t_py=0.0, t_js=0.0, count_ok=0)
with tempfile.TemporaryDirectory() as tmp:
    for v in sys.argv[1:]:
        # the network alone, on identical frames
        cap, fr = cv2.VideoCapture(v), []
        while len(fr) < N:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(cv2.resize(f, (100, 100)))
        cap.release()
        raw, out = os.path.join(tmp, "frames.raw"), os.path.join(tmp, "p.npy")
        np.stack(fr).tofile(raw)
        subprocess.run(["node", RAW, raw, out], check=True)
        dm = float(np.abs(np.load(out) - leanscene.frame_probs(fr, model)).max())
        # end to end
        t = time.perf_counter()
        py = leanscene.frame_probs(v, model)
        t_py = time.perf_counter() - t
        t = time.perf_counter()
        subprocess.run(["node", CLI, v, "-q", "-f", "json", "-o", os.path.join(tmp, "r.json"), "--probs", out], check=True)
        t_js = time.perf_counter() - t
        js = np.load(out)
        same = len(js) == len(py)
        n = min(len(js), len(py))
        d = np.abs(js[:n] - py[:n])
        ep, ej = events(py), events(js)
        r = json.load(open(os.path.join(tmp, "r.json")))
        assert [(x["start"], x["end"]) for x in r["transitions"]] == ej, "the CLI's transitions differ from the rule applied to its probabilities"
        tot["frames"] += len(py); tot["n_py"] += len(ep); tot["n_js"] += len(ej)
        tot["only_py"] += unmatched(ep, ej); tot["only_js"] += unmatched(ej, ep)
        tot["dp"].append(float(d.mean())); tot["dpmax"] = max(tot["dpmax"], float(d.max())); tot["dm"] = max(tot["dm"], dm)
        tot["t_py"] += t_py; tot["t_js"] += t_js; tot["count_ok"] += same
        name = os.path.basename(v)
        print(f"{os.path.basename(os.path.dirname(v)) if name == 'video.mp4' else name}: frames {len(py)} / {len(js)}{'' if same else '  DIFFERENT'}, "
              f"transitions {len(ep)} / {len(ej)}, only Python {unmatched(ep, ej)}, only npm {unmatched(ej, ep)}, "
              f"probability difference mean {d.mean():.4f} max {d.max():.2f}, model alone max {dm:.1e}, "
              f"{len(py) / t_py:.0f} / {len(js) / t_js:.0f} frames/s", flush=True)
k = len(sys.argv) - 1
print(f"\n{k} videos, {tot['frames']:,} frames. Same frame count on {tot['count_ok']} of {k}.")
print(f"Network alone, identical frames: largest probability difference {tot['dm']:.1e}.")
print(f"End to end: Python finds {tot['n_py']} transitions, npm {tot['n_js']}; {tot['only_py']} only in Python, {tot['only_js']} only in npm "
      f"(two-frame tolerance). Probability difference: mean {np.mean(tot['dp']):.4f}, largest {tot['dpmax']:.2f}.")
print(f"Speed end to end, CPU: Python {tot['frames'] / tot['t_py']:.0f} frames/s, npm {tot['frames'] / tot['t_js']:.0f} frames/s.")
