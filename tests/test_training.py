"""The training code in training/: targets, the loss, and one synthetic batch through the model."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import clipshots  # noqa: E402
import hardneg  # noqa: E402
import labels  # noqa: E402
import train  # noqa: E402

from leanscene import LeanScene  # noqa: E402


def test_label_targets():
    t = labels.targets(100, 40, [(120, "cut"), (130, "same")])
    assert t[20] == 1  # the cut frame
    assert t[19] == labels.IGNORE and t[21] == labels.IGNORE  # one frame either side is not supervised
    assert (t[12:19] == 0).all() and (t[22:29] == 0).all()  # f±2..f±8 count as same
    assert (t[27:34] == 0).all()  # same at 130: 127..133
    assert t[0] == labels.IGNORE and t[39] == labels.IGNORE  # unchecked frames


def test_clipshots_targets():
    t = clipshots.targets(50, [[9, 10], [20, 26]], exhaustive=True)
    assert t[10] == 1 and t[9] == clipshots.IGNORE and t[11] == clipshots.IGNORE
    assert (t[21:26] == 2).all() and t[20] == 0 and t[26] == 0
    t = clipshots.targets(50, [[20, 26]], exhaustive=False)  # only_gradual: cuts are unlabeled
    assert (t[21:26] == 2).all() and (t[18:21] == 0).all() and (t[26:29] == 0).all()
    assert t[0] == clipshots.IGNORE and t[40] == clipshots.IGNORE


def test_false_alarms_skip_annotated_transitions():
    pr = np.zeros((200, 3), np.float32)
    pr[:, 0] = 1
    pr[50, 1] = 0.9   # at an annotated cut
    pr[120, 1] = 0.8  # nowhere near one
    pr[122, 2] = 0.6  # same event as 120
    t = np.zeros(200, np.int64)
    t[50] = 1
    events = hardneg.false_alarms(pr, t)
    assert [f for _, f in events] == [120]


def test_loss_is_weighted_cross_entropy_over_labeled_frames():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 64, 3, generator=g)
    y = torch.randint(0, 3, (3, 64), generator=g)
    y[torch.rand(3, 64, generator=g) < 0.3] = -100
    w = torch.tensor([1.0, 4.0, 2.0])
    want = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), weight=w, ignore_index=-100)
    assert torch.allclose(train.loss_fn(logits, y, w), want, atol=1e-6)


def test_synthetic_batch_trains_one_step():
    rs = np.random.RandomState(0)
    clips = [[np.full((100, 100, 3), rs.randint(0, 255, 3), np.uint8) + rs.randint(0, 8, (100, 100, 3), dtype=np.uint8)
              for _ in range(n)] for n in (90, 70, 40, 30, 80)]
    clips = train.ClipSet(clips, ["a", "a", "b", "c", "c"], [0, 500, 0, 0, 90])
    x, y = train.batch(clips, random.Random(0), 4, dirty_cut=0.3, dirty_same=0.12)
    assert x.shape == (4, 64, 3, 100, 100) and y.shape == (4, 64)
    assert set(np.unique(y.numpy())) <= {0, 1, 2}
    model = LeanScene(width=0.5)
    loss = train.loss_fn(model(x), y, torch.tensor([1.0, 4.0, 2.0]))
    loss.backward()
    assert torch.isfinite(loss)
