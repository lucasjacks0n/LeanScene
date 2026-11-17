"""Export bundled LeanScene weights for the browser demo (demo/index.html).

    python demo/export_onnx.py [checkpoint.pt]

Needs the export extras: pip install -e ".[export]". Writes demo/model.js (about 2.7 MB): two
ONNX graphs, base64 in a script file, so the page also works opened straight from disk
(file://), where fetch() of a model file is blocked.

  features: (N,3,100,100) float32 BGR in [0,1] -> (N,D-64) per-frame features, without the
            colour histogram, which the page computes itself and inserts at column 128
  head:     (1,S,D) features of a whole video    -> (S,3) softmax same/cut/fade

Then checks the two graphs, assembled as the page assembles them, against PyTorch on random
frames and prints the largest probability difference.
"""
import base64, io, json, os, sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

import leanscene
from leanscene.model import FRAME_SIZE, color_hist

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "leanscene"
m = leanscene.load_model(name, "cpu")
assert m.spatial and m.feat == 128, "the demo page covers the default architecture only"
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


def export(mod, x, axis):
    buf = io.BytesIO()
    torch.onnx.export(mod.eval(), (x,), buf, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {axis: "n"}, "y": {0: "n"}}, opset_version=17, dynamo=False)
    return buf.getvalue()


torch.manual_seed(0)
x = torch.rand(7, 3, S, S)
check = torch.rand(300, 3, S, S)  # a different length than the export example
with torch.no_grad():
    ref = torch.softmax(m.head(m.frame_features(check)[None])[0], 1).numpy()
    f = m.frame_features(x)
    feats, head = export(Features(), x, 0), export(Head(), f[None], 1)
meta = {"checkpoint": os.path.basename(str(name)), "dim": int(f.shape[1]), "size": S, "cut": leanscene.CUT_THRESHOLD,
        "fade": leanscene.FADE_THRESHOLD, "close": leanscene.FADE_CLOSE}
with open(os.path.join(HERE, "model.js"), "w") as fh:
    fh.write("window.LEANSCENE = " + json.dumps({"meta": meta, "features": base64.b64encode(feats).decode(),
                                                 "head": base64.b64encode(head).decode()}) + ";\n")
print(f"wrote demo/model.js: features {len(feats) / 1e6:.2f} MB, head {len(head) / 1e6:.2f} MB, feature dim {meta['dim']}")

sf, sh = ort.InferenceSession(feats), ort.InferenceSession(head)
ff = np.concatenate([sf.run(None, {"x": check[i:i + 16].numpy()})[0] for i in range(0, 300, 16)])
ff = np.concatenate([ff[:, :m.feat], color_hist(check).numpy(), ff[:, m.feat:]], 1)  # as the page assembles it
diff = float(np.abs(sh.run(None, {"x": ff[None]})[0] - ref).max())
print("onnxruntime vs PyTorch, 300 random frames: max probability difference", diff)
assert diff < 1e-4
