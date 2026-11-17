import json

import cv2
import numpy as np
import pytest
import torch

import leanscene
from leanscene.cli import main


def shot(seed, n, size=(64, 96)):
    """n frames of one smooth random image drifting slowly sideways."""
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 256, (6, 9, 3), dtype=np.uint8), (size[1] + n, size[0]), interpolation=cv2.INTER_CUBIC)
    return [np.ascontiguousarray(base[:, i:i + size[1]]) for i in range(n)]


def video():
    """Three shots, 130 frames: a hard cut at frame 40, then a fade through black over frames 80-94."""
    a, b, c = shot(1, 40), shot(2, 46), shot(3, 41)
    out = [(f * (1 - (i + 1) / 7)).astype(np.uint8) for i, f in enumerate(b[40:])]
    black = [np.zeros_like(a[0])] * 3
    into = [(f * ((i + 1) / 7)).astype(np.uint8) for i, f in enumerate(c[:6])]
    return a + b[:40] + out + black + into + c[6:]


@pytest.fixture(scope="module")
def model():
    return leanscene.load_model(device="cpu")


@pytest.mark.parametrize("name", sorted(leanscene.MODELS))
def test_bundled_weights_load(name):
    m = leanscene.load_model(name, device="cpu")
    assert sum(p.numel() for p in m.parameters()) == 489_243  # the paper's parameter count
    assert m.receptive_field == 63


def test_unknown_model_name():
    with pytest.raises(ValueError):
        leanscene.load_model("no-such-model")


def test_finds_the_cut_and_the_fade(model):
    frames = video()
    r = leanscene.detect(frames, model=model)
    assert r.frames == len(frames) and r.probs.shape == (len(frames), 3) and r.fps is None
    assert [t.kind for t in r.transitions] == ["cut", "gradual"]
    cut, fade = r.transitions
    assert (cut.start, cut.end) == (40, 40)
    assert 80 <= fade.start <= fade.end <= 94  # inside the fade
    assert len(r.shots) == 3 and r.shots[0] == (0, 39) and r.shots[1][0] == 40


def test_streaming_matches_a_single_pass(model):
    frames = [cv2.resize(f, (100, 100)) for f in video() * 3]
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255)
    with torch.inference_mode():
        whole = torch.softmax(model(x[None])[0], 1).numpy()
    streamed = leanscene.frame_probs(frames, model=model, batch=7, window=50)
    assert np.abs(streamed - whole).max() < 1e-4


def test_empty_source(model):
    r = leanscene.detect([], model=model)
    assert r.frames == 0 and r.shots == [] and r.transitions == []


def test_rejects_frames_that_are_not_bgr_uint8(model):
    with pytest.raises(ValueError):
        leanscene.detect([np.zeros((10, 10), np.uint8)], model=model)


def test_cli_on_a_video_file(tmp_path, capsys):
    path = str(tmp_path / "clip.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 64))
    if not w.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    for f in video():
        w.write(f)
    w.release()

    assert main([path, "--format", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["frames"] == 130 and d["fps"] == 25
    assert d["transitions"][0] == {"start": 40, "end": 40, "kind": "cut", "start_time": 1.6, "end_time": 1.64}
    assert d["shots"][0] == {"start": 0, "end": 39, "start_time": 0.0, "end_time": 1.6}

    assert main([path]) == 0
    assert "3 shots" in capsys.readouterr().out
    assert main([str(tmp_path / "missing.mp4")]) == 1
    assert "no such file" in capsys.readouterr().err
