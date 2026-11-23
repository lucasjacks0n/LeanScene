"""Export the bundled LeanScene weights for the npm package (js/weights/) and the Python
package's ONNX Runtime backend (src/leanscene/weights/, the same files).

    python js/scripts/export_onnx.py [checkpoint.pt]

Needs the export extras: pip install -e ".[export]". Writes three files, about 1.6 MB together:

  features.onnx  (N,3,100,100) float32 BGR in [0,1] -> (N,D-64) per-frame features, without the
                 color histogram, which the package computes itself and inserts at column 128
  head.onnx      (1,S,D) features of a run of frames -> (S,3) softmax same / cut / fade
  meta.json      feature size, frame size, the decision rule, and the checkpoint's name

Then checks the two graphs, assembled as the package assembles them, against PyTorch on random
frames and prints the largest probability difference.
"""
import json, os, sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

import leanscene
from leanscene.model import FRAME_SIZE, HIST_BINS, color_hist

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
name = sys.argv[1] if len(sys.argv) > 1 else "leanscene"
m = leanscene.load_model(name, "cpu", backend="torch")  # exports from the PyTorch module
assert m.spatial and m.feat == 128, "the package covers the default architecture only"
S = FRAME_SIZE
with torch.no_grad():
    FM = int(m.enc(torch.zeros(1, 3, S, S)).shape[2])  # side of the CNN feature map


class Features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = m  # registered, so the weights are exported as parameters

    def forward(self, x):
        # LeanScene.frame_features with the pooling sizes as plain ints for the fixed input size:
        # traced from the real method they become tensors, which avg_pool2d cannot export.
        fmap = m.enc(x)
        h, fh = S // 8 * 8, FM // 2 * 2
        xc = x[:, :, (S - h) // 2:(S - h) // 2 + h, (S - h) // 2:(S - h) // 2 + h]
        return torch.cat([fmap.mean((2, 3)), F.avg_pool2d(xc, (h // 8, h // 8)).flatten(1),
                          F.avg_pool2d(fmap[:, :, :fh, :fh], (fh // 2, fh // 2)).flatten(1)], 1)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = m

    def forward(self, f):
        return torch.softmax(m.head(f)[0], 1)


def export(mod, x, axis, path):
    torch.onnx.export(mod.eval(), (x,), path, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {axis: "n"}, "y": {0: "n"}}, opset_version=17, dynamo=False)
    return os.path.getsize(path)


torch.manual_seed(0)
x = torch.rand(7, 3, S, S)
check = torch.rand(300, 3, S, S)  # a different length than the export example
os.makedirs(OUT, exist_ok=True)
fp, hp = os.path.join(OUT, "features.onnx"), os.path.join(OUT, "head.onnx")
with torch.no_grad():
    ref = torch.softmax(m.head(m.frame_features(check)[None])[0], 1).numpy()
    f = m.frame_features(x)
    nf, nh = export(Features(), x, 0, fp), export(Head(), f[None], 1, hp)
meta = {"checkpoint": os.path.basename(str(name)), "dim": int(f.shape[1]), "feat": m.feat, "size": S,
        "histBins": HIST_BINS, "receptiveField": m.receptive_field, "cut": leanscene.CUT_THRESHOLD,
        "fade": leanscene.FADE_THRESHOLD, "close": leanscene.FADE_CLOSE}
with open(os.path.join(OUT, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
    fh.write("\n")
print(f"wrote js/weights: features.onnx {nf / 1e6:.2f} MB, head.onnx {nh / 1e6:.2f} MB, feature dim {meta['dim']}")
# The Python package runs the same graphs on ONNX Runtime by default (leanscene.fast): same files.
import shutil
PY = os.path.join(os.path.dirname(os.path.dirname(OUT)), "src", "leanscene", "weights")
for f in ("features.onnx", "head.onnx", "meta.json"):
    shutil.copyfile(os.path.join(OUT, f), os.path.join(PY, f))
print("copied them to src/leanscene/weights")

sf, sh = ort.InferenceSession(fp), ort.InferenceSession(hp)
ff = np.concatenate([sf.run(None, {"x": check[i:i + 16].numpy()})[0] for i in range(0, 300, 16)])
ff = np.concatenate([ff[:, :m.feat], color_hist(check).numpy(), ff[:, m.feat:]], 1)  # as the package assembles it
diff = float(np.abs(sh.run(None, {"x": ff[None]})[0] - ref).max())
print("onnxruntime vs PyTorch, 300 random frames: max probability difference", diff)
assert diff < 1e-4
