"""Synthetic transitions and same-shot hard cases applied to real clip frames.

train.py builds its synthetic training sequences from these: single-shot clips are spliced
with cuts, dissolves and dips, and same-shot disturbances (flashes, exposure swings, whip pans,
lens flares, occlusions) are added as negatives. All functions take and return lists of uint8
BGR frames and work at any resolution.
"""

from pathlib import Path

import cv2
import numpy as np

# Clips are named <source video id>-<start frame>-<frame count>.mp4. The source id is what
# train.py balances its sampling over, so a video with many clips is not drawn more often.


def clip_source(path):
    return Path(path).stem.rsplit("-", 2)[0]


def clip_paths(directory):
    """Every .mp4 in `directory`, sorted by name."""
    return sorted(Path(directory).glob("*.mp4"))


def read_clip(path, size):
    """size: (w, h)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


# ---------- transitions (a = outgoing frames, b = incoming frames) ----------


def cut(a, b):
    """Returns (frames, boundary index = first frame of b)."""
    return a + b, len(a)


def fade(a, b, L, alphas=None):
    """Cross-fade with L blended frames (alpha = i/(L+1) unless given), replacing
    a's last L frames and b's first L frames. Returns (frames, middle blended index)."""
    if alphas is None:
        alphas = [(i + 1) / (L + 1) for i in range(L)]
    blend = [cv2.addWeighted(a[len(a) - L + i], 1 - alphas[i], b[i], alphas[i], 0) for i in range(L)]
    first = len(a) - L
    return a[:first] + blend + b[L:], first + L // 2


def dip_to_black(a, b, out_len, hold, in_len):
    """a fades to black over out_len frames, holds black, b fades in over in_len."""
    out = [(a[len(a) - out_len + i] * (1 - (i + 1) / (out_len + 1))).astype(np.uint8) for i in range(out_len)]
    black = [np.zeros_like(a[0]) for _ in range(hold)]
    inn = [(b[i] * ((i + 1) / (in_len + 1))).astype(np.uint8) for i in range(in_len)]
    mid = out + black + inn
    first = len(a) - out_len
    return a[:first] + mid + b[in_len:], first + len(mid) // 2


def dip_to_white(a, b, out_len, hold, in_len, rng=None):
    """Flash cut: a blows out to white over out_len frames, holds near-white
    (a faint remnant of the last frame), then b fades in from white over in_len.
    Unlike the `flash` hard case, the shot changes across the white, so it's a
    transition (labeled like a fade)."""
    remnant = rng.uniform(0.0, 0.15) if rng else 0.08
    white = np.full_like(a[0], 255)
    out = [cv2.addWeighted(a[len(a) - out_len + i], 1 - (i + 1) / (out_len + 1), white, (i + 1) / (out_len + 1), 0)
           for i in range(out_len)]
    hold_frames = [cv2.addWeighted(a[-1], remnant, white, 1 - remnant, 0) for _ in range(hold)]
    inn = [cv2.addWeighted(b[i], (i + 1) / (in_len + 1), white, 1 - (i + 1) / (in_len + 1), 0) for i in range(in_len)]
    mid = out + hold_frames + inn
    first = len(a) - out_len
    return a[:first] + mid + b[in_len:], first + len(mid) // 2


# ---------- same-shot hard cases (should NOT be detected) ----------


def flash(frames, at, rng):
    """1-2 frame camera flash / blown-out frame."""
    f = [x.copy() for x in frames]
    n = rng.choice([1, 1, 2])
    s = rng.uniform(0.5, 1.0)
    for i in range(at, min(at + n, len(f))):
        f[i] = cv2.addWeighted(f[i], 1 - s, np.full_like(f[i], 255), s, 0)
    return f, (at, min(at + n, len(f)) - 1)


def exposure_swing(frames, at, rng):
    """Sun-flare style exposure ramp: brightness and a glare blob rise and fall
    over 4-12 frames."""
    f = [x.copy() for x in frames]
    n = rng.randint(4, 12)
    peak = rng.uniform(1.6, 3.0)
    h, w = f[0].shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h * 0.6)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rng.uniform(0.15, 0.4) * w) ** 2)))[..., None]
    end = min(at + n, len(f))
    for i in range(at, end):
        t = np.sin(np.pi * (i - at + 1) / (n + 1))  # 0 -> 1 -> 0
        g = 1 + (peak - 1) * t
        x = f[i].astype(np.float32) * g + 255 * 0.8 * t * blob
        f[i] = np.clip(x, 0, 255).astype(np.uint8)
    return f, (at, end - 1)


def whip_pan(frames, at, rng):
    """Fast camera move: frames slide hard in one direction with motion blur
    over 3-8 frames, then settle back."""
    f = [x.copy() for x in frames]
    n = rng.randint(3, 8)
    h, w = f[0].shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(ang), np.sin(ang)
    end = min(at + n, len(f))
    for i in range(at, end):
        t = np.sin(np.pi * (i - at + 1) / (n + 1))
        shift = t * rng.uniform(0.15, 0.35) * w
        M = np.float32([[1, 0, dx * shift], [0, 1, dy * shift]])
        x = cv2.warpAffine(f[i], M, (w, h), borderMode=cv2.BORDER_REFLECT)
        k = max(1, int(t * 0.12 * w))
        if k > 1:
            ker = np.zeros((k, k), np.float32)
            c = k // 2
            cv2.line(ker, (int(c - dx * c), int(c - dy * c)), (int(c + dx * c), int(c + dy * c)), 1.0, 1)
            x = cv2.filter2D(x, -1, ker / ker.sum())
        f[i] = x
    return f, (at, end - 1)


def flare_flicker(frames, at, rng):
    """Fisheye sun flare as the camera moves through the sun: a large glare blob
    whose strength jumps abruptly frame to frame over 3-10 frames."""
    f = [x.copy() for x in frames]
    n = rng.randint(3, 10)
    h, w = f[0].shape[:2]
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0, 0.7) * h
    vx, vy = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.04, 0.04) * h
    yy, xx = np.mgrid[0:h, 0:w]
    rad = rng.uniform(0.2, 0.5) * w
    end = min(at + n, len(f))
    for k, i in enumerate(range(at, end)):
        s = rng.choice([0.0, rng.uniform(0.3, 1.0), rng.uniform(0.6, 1.0)])
        x0, y0 = cx + vx * k, cy + vy * k
        blob = np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * rad**2)))[..., None]
        tint = np.array([rng.uniform(0.7, 1.0), rng.uniform(0.85, 1.0), 1.0]) * 255  # BGR, warm
        x = f[i].astype(np.float32) * (1 + 0.3 * s * blob) + 0.55 * s * blob * tint
        f[i] = np.clip(x, 0, 255).astype(np.uint8)
    return f, (at, end - 1)


def occlusion(frames, at, rng):
    """Foreground object (pillar, wall, person) sweeps across the lens over
    3-8 frames: a blurred flat-colored region enters and leaves."""
    f = [x.copy() for x in frames]
    n = rng.randint(3, 8)
    h, w = f[0].shape[:2]
    color = np.array([rng.randint(20, 200)] * 3, np.float32) + np.array([rng.randint(-20, 20) for _ in range(3)])
    width = rng.uniform(0.4, 1.2) * w
    horizontal = rng.random() < 0.8
    span = (w if horizontal else h) + width
    end = min(at + n, len(f))
    for k, i in enumerate(range(at, end)):
        pos = -width + span * (k + 1) / (n + 1)
        mask = np.zeros((h, w), np.float32)
        if horizontal:
            a, b = int(max(0, pos)), int(min(w, pos + width))
            mask[:, a:b] = 1
        else:
            a, b = int(max(0, pos)), int(min(h, pos + width))
            mask[a:b, :] = 1
        k_ = max(3, int(0.08 * w)) | 1
        mask = cv2.GaussianBlur(mask, (k_, k_), 0)[..., None]
        tex = color + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, (h, w, 3))
        f[i] = np.clip(f[i] * (1 - mask) + tex * mask, 0, 255).astype(np.uint8)
    return f, (at, end - 1)


DIRTY_KINDS = ["leak", "blur", "white", "tint", "dark", "occluder"]


def dirty_tail(frames, k, rng, carry=None, kinds=None):
    """Disturbance that builds over the last k frames of an outgoing shot and is cut
    off by the cut instead of recovering (light leak, blur, whiteout, tint, darkening,
    an occluder moving in). carry: incoming-shot frames that get the same disturbance
    fading out over 1-2 frames. kinds: subset of DIRTY_KINDS to pick from.
    Returns (frames, carry frames)."""
    f = [x.copy() for x in frames]
    h, w = f[0].shape[:2]
    kind = rng.choice(kinds or DIRTY_KINDS)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rng.uniform(0.3, 0.7) * w) ** 2)))[..., None]
    tint = np.array([rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)], np.float32) * 255
    if kind == "leak":
        tint = np.array([rng.uniform(0.2, 0.7), rng.uniform(0.5, 0.9), 1.0], np.float32) * 255  # warm BGR
    peak = rng.uniform(0.5, 1.0)
    horizontal, from_start = rng.random() < 0.7, rng.random() < 0.5

    def apply(x, t):  # t in (0, 1]: strength
        x = x.astype(np.float32)
        if kind == "leak":
            a = 0.9 * t * blob
            x = x * (1 - a) + tint * a
        elif kind == "blur":
            r = int(t * 0.15 * w) | 1
            if r > 1:
                x = cv2.GaussianBlur(x, (r, r), 0)
            x = x * (1 - 0.3 * t) + 255 * 0.3 * t
        elif kind == "white":
            x = x * (1 + 1.5 * t) + 255 * 0.7 * t
        elif kind == "tint":
            x = x * (1 - 0.6 * t) + tint * 0.6 * t
        elif kind == "dark":
            x = x * (1 - 0.92 * t)
        else:
            size = (w if horizontal else h)
            edge = int(size * t)
            m = np.zeros((h, w), np.float32)
            if horizontal:
                if from_start:
                    m[:, :edge] = 1
                else:
                    m[:, w - edge:] = 1
            else:
                if from_start:
                    m[:edge, :] = 1
                else:
                    m[h - edge:, :] = 1
            kk = max(3, int(0.08 * w)) | 1
            m = cv2.GaussianBlur(m, (kk, kk), 0)[..., None]
            x = x * (1 - m) + tint.mean() * 0.4 * m
        return np.clip(x, 0, 255).astype(np.uint8)

    n = len(f)
    for j, i in enumerate(range(n - k, n)):
        t = peak * ((j + 1) / k) ** rng.uniform(0.5, 1.5)
        if kind == "leak" or kind == "white":
            t = min(1.0, t * rng.uniform(0.7, 1.3))  # leaks flicker
        f[i] = apply(f[i], t)
    c = [x.copy() for x in carry] if carry else []
    for j in range(len(c)):
        c[j] = apply(c[j], peak * (0.6 if j == 0 else 0.3))
    return f, c


HARD_CASES = {
    "flash": flash,
    "exposure": exposure_swing,
    "whip": whip_pan,
    "flare": flare_flicker,
    "occlusion": occlusion,
}
