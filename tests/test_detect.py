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
    return leanscene.load_model(device="cpu", backend="torch")


@pytest.mark.parametrize("name", sorted(leanscene.MODELS))
def test_bundled_weights_load(name):
    m = leanscene.load_model(name, device="cpu", backend="torch")
    assert sum(p.numel() for p in m.parameters()) == 394_575  # the paper's half-width model
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


def test_progress_reports_every_frame(model):
    seen = []
    r = leanscene.detect(video(), model=model, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == r.frames == 130


# ---- ONNX Runtime, the default on the CPU, against PyTorch

onnx = pytest.mark.skipif(not leanscene.fast.available(), reason="onnxruntime is not installed")


@pytest.fixture(scope="module")
def onnx_model():
    return leanscene.load_model(device="cpu", backend="onnx")


def write_video(path, frames, fps=25):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    if not w.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    for f in frames:
        w.write(f)
    w.release()
    return str(path)


@onnx
def test_the_default_on_the_cpu_is_onnx_runtime():
    if torch.cuda.is_available():
        pytest.skip("CUDA is the default device here")
    assert isinstance(leanscene.load_model(), leanscene.OnnxModel)
    assert isinstance(leanscene.load_model(device="cpu"), leanscene.OnnxModel)


@onnx
def test_onnx_gives_the_same_probabilities_as_pytorch(model, onnx_model):
    frames = [cv2.resize(f, (100, 100)) for f in video() * 3]
    a = leanscene.frame_probs(frames, model=model)
    b = leanscene.frame_probs(frames, model=onnx_model, batch=7, window=50)
    assert np.abs(a - b).max() < 1e-4
    assert leanscene.find_transitions(leanscene.label_frames(a)) == leanscene.find_transitions(leanscene.label_frames(b))


def test_the_fast_histogram_equals_the_model_s():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (6, 100, 100, 3), dtype=np.uint8)
    frames[0] = np.tile(np.arange(256, dtype=np.uint8), 118)[:30000].reshape(100, 100, 3)  # every byte value
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255)
    assert np.array_equal(leanscene.fast.color_hist(frames), leanscene.model.color_hist(x).numpy())


@onnx
def test_a_file_decoded_in_a_thread_gives_what_frames_in_this_thread_give(tmp_path, model, onnx_model):
    path = write_video(tmp_path / "clip.mp4", video())
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    for m in (model, onnx_model):
        assert np.array_equal(leanscene.frame_probs(path, model=m), leanscene.frame_probs(frames, model=m))


@onnx
def test_errors_in_the_decoder_thread_reach_the_caller(tmp_path, onnx_model):
    with pytest.raises(FileNotFoundError):
        leanscene.detect(str(tmp_path / "missing.mp4"), model=onnx_model)
    not_a_video = tmp_path / "x.mp4"
    not_a_video.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        leanscene.detect(str(not_a_video), model=onnx_model)


def test_the_decoder_thread_stops_when_the_caller_stops():
    import threading

    def frames():
        for i in range(10_000):
            yield np.zeros((100, 100, 3), np.uint8)

    import importlib

    gen = importlib.import_module("leanscene.detect")._prefetch(frames(), batch=4, depth=2)
    next(gen)
    gen.close()  # the caller gives up early
    assert not any(t.name == "leanscene-decoder" and t.is_alive() for t in threading.enumerate())


@onnx
def test_backend_choices(onnx_model):
    with pytest.raises(ValueError):
        leanscene.load_model(__file__, backend="onnx")  # a checkpoint file: PyTorch only
    with pytest.raises(ValueError):
        leanscene.load_model(device="mps", backend="onnx")
    with pytest.raises(ValueError):
        leanscene.load_model(backend="tensorflow")
    with pytest.raises(ValueError):
        leanscene.frame_probs([], model=onnx_model, device="mps")


def test_the_onnx_weights_are_the_npm_package_s():
    import pathlib
    from importlib import resources

    js = pathlib.Path(__file__).resolve().parent.parent / "js" / "weights"
    if not js.exists():
        pytest.skip("not in the repository")
    here = resources.files("leanscene") / "weights"
    for name in ("features.onnx", "head.onnx", "meta.json"):
        assert (here / name).read_bytes() == (js / name).read_bytes(), name
