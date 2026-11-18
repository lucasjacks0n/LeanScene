"""Train LeanScene.

Each training sequence is 64 frames at 100x100 with a label per frame: 0 = same, 1 = cut (the
first frame of the new shot), 2 = fade (a frame inside a gradual transition). A sequence comes
from one of three sources:

  synthetic   single-shot clips (--clips) spliced with cuts, dissolves and dips, plus
              same-shot disturbances as negatives (synth.py)
  --real      windows around the hand labels in labels/labels.csv (labels.py)
  --clipshots windows from ClipShots training videos (clipshots.py), part of them mined
              false alarms (hardneg.py)

The command for the paper's model, and what each data source needs: training/README.md.

    python training/train.py --clips <dir of single-shot clips> --out leanscene.pt --spatial --width 0.5 ...
"""

import argparse, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth
import labels
import clipshots

from leanscene import LeanScene

HARD_WEIGHTS = {"flash": 0.08, "exposure": 0.2, "whip": 0.24, "flare": 0.24, "occlusion": 0.24}
FADE_LENS = [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 9, 11, 15, 20]


# ---------------- data ----------------


class ClipSet(list):
    """Clip frames plus each clip's source video, so sampling can weight
    source videos equally instead of favoring the video with the most clips."""

    def __init__(self, clips, sources, starts=None):
        super().__init__(clips)
        self.starts = starts or [None] * len(clips)  # each clip's start frame in its source video
        self.groups = {}
        for i, s in enumerate(sources):
            self.groups.setdefault(s, []).append(i)
        self.group_list = list(self.groups.values())


def pick(clips, rng):
    """Index of a random clip: uniform over source videos when clips is a ClipSet."""
    if isinstance(clips, ClipSet):
        return rng.choice(rng.choice(clips.group_list))
    return rng.randrange(len(clips))


DIRTY_KINDS = None  # --dirty-kinds; None = all of synth.DIRTY_KINDS
DIRTY_SAME_WHIP = 0.0  # --dirty-same-whip
CS_HARDNEG, CS_HARDNEG_P = None, 0.0  # --clipshots-hardneg (hardneg.py windows)
CS_FADENEG, CS_FADENEG_P = None, 0.0  # --clipshots-fadeneg (fade false alarms, HARDNEG_DIR=clipshots_fadeneg)
CS_GRADUAL, CS_GRADUAL_P = None, 0.0  # --clipshots-gradual (the ClipShots windows that hold a gradual transition)
SAME_SOURCE_NEAR = 0  # --same-source-near: same-source shots come from within this many source frames


def make_sequence(clips, rng, S=64, same_source=0.0, junk=0.0, dirty_cut=0.0, dirty_same=0.0):
    """Concatenate random shots with random transitions. Returns frames (S,H,W,3)
    and per-frame labels (S,). same_source: chance the next shot comes from the same
    source video as the previous one (cuts between similar-looking shots). junk: chance
    a hard cut is preceded by 1-2 frames of an unrelated clip (both edges labeled cut).
    dirty_cut: chance a hard cut's outgoing shot ends in a disturbance that the cut cuts
    off (synth.dirty_tail); when > 0, 8% of shots are also only 2-6 frames long.
    dirty_same: chance per shot of the matched negative, the same disturbance building
    inside a shot and then dropping away to the same shot (labeled same), so "a
    disturbance ends abruptly" alone never means cut."""

    def dirty_inside(seg):
        if len(seg) < 6 or rng.random() >= dirty_same:
            return seg
        if rng.random() < DIRTY_SAME_WHIP:
            return synth.whip_pan(list(seg), rng.randint(0, len(seg) - 3), rng)[0]
        k = rng.randint(1, min(8, len(seg) - 3))
        at = rng.randint(0, len(seg) - k - 1)
        c = min(rng.choice([0, 0, 1, 2]), len(seg) - at - k)
        mid, after = synth.dirty_tail(list(seg[at:at + k]), k, rng, list(seg[at + k:at + k + c]), DIRTY_KINDS)
        return list(seg[:at]) + mid + after + list(seg[at + k + c:])

    if rng.random() < 0.12:  # a single shot, maybe with a hard case
        while True:
            c = clips[pick(clips, rng)]
            if len(c) >= S:
                break
        s = rng.randint(0, len(c) - S)
        seq = c[s : s + S]
        if rng.random() < 0.7:
            name = rng.choices(list(HARD_WEIGHTS), weights=list(HARD_WEIGHTS.values()))[0]
            seq, _ = synth.HARD_CASES[name](seq, rng.randint(0, S - 4), rng)
        seq = dirty_inside(seq)
        return seq, np.zeros(S, np.int64)

    out, lab, prev = [], [], None
    while len(out) < S + 2:
        sib = same_source > 0 and isinstance(clips, ClipSet) and prev is not None and rng.random() < same_source
        group = next((g for g in clips.group_list if prev in g), None) if sib else None
        if group is not None and len(group) < 2:
            group = None
        if group and SAME_SOURCE_NEAR and clips.starts[prev] is not None:  # same session/spot: nearby in the source
            near = [i for i in group if i != prev and clips.starts[i] is not None
                    and abs(clips.starts[i] - clips.starts[prev]) <= SAME_SOURCE_NEAR]
            group = near or group
        while True:
            ci = rng.choice(group) if group else pick(clips, rng)
            n = rng.randint(2, 6) if dirty_cut > 0 and rng.random() < 0.08 else rng.randint(8, 40)
            if ci != prev and len(clips[ci]) >= n:
                break
        prev = ci
        st = rng.randint(0, len(clips[ci]) - n)
        seg = clips[ci][st : st + n]
        if rng.random() < 0.4:
            name = rng.choices(list(HARD_WEIGHTS), weights=list(HARD_WEIGHTS.values()))[0]
            seg, _ = synth.HARD_CASES[name](seg, rng.randint(0, max(0, n - 3)), rng)
        seg = dirty_inside(seg)
        if not out:
            out, lab = list(seg), [0] * n
            continue
        r = rng.random()
        if r < 0.35:
            if junk > 0 and rng.random() < junk:
                jc = clips[pick(clips, rng)]
                k = rng.randint(1, 2)
                j0 = rng.randint(0, len(jc) - k)
                lab += [1] + [0] * (k - 1)
                out = list(out) + list(jc[j0 : j0 + k])
            if dirty_cut > 0 and rng.random() < dirty_cut:
                clean = 0  # frames since the last transition
                while clean < len(lab) and lab[-1 - clean] == 0:
                    clean += 1
                k = min(rng.randint(1, 8), clean - 1)
                if k >= 1:
                    c = rng.choice([0, 0, 1, 2]) if n > 3 else 0
                    tail, head = synth.dirty_tail(list(out[-k:]), k, rng, list(seg[:c]), DIRTY_KINDS)
                    out = list(out[:-k]) + tail
                    seg = head + list(seg[c:])
            b = len(out)
            out, _ = synth.cut(out, list(seg))
            lab += [0] * n
            lab[b] = 1
        elif r < 0.8:
            L = min(rng.choice(FADE_LENS), n - 2, len(out) - 2)
            if L < 1:
                out, _ = synth.cut(out, list(seg)); lab += [0] * n; lab[len(out) - n] = 1
                continue
            al = None
            if L <= 4 and rng.random() < 0.7:
                al = sorted(min(0.9, max(0.1, (i + 1) / (L + 1) + rng.uniform(-0.2, 0.2))) for i in range(L))
            first = len(out) - L
            out, _ = synth.fade(out, list(seg), L, al)
            lab = lab[:first] + [2] * L + [0] * (n - L)
        else:
            o, hd, i = rng.randint(1, 3), rng.randint(0, 4), rng.randint(1, 3)
            if n <= i + 1 or len(out) <= o + 1:
                continue
            first = len(out) - o
            if rng.random() < 0.5:  # flash cut through white (labeled as a transition, unlike an in-shot flash)
                out, _ = synth.dip_to_white(out, list(seg), o, hd, i, rng)
            else:
                out, _ = synth.dip_to_black(out, list(seg), o, hd, i)
            lab = lab[:first] + [2] * (o + hd + i) + [0] * (n - i)
        assert len(out) == len(lab), (len(out), len(lab))
    s = rng.randint(0, len(out) - S)
    return out[s : s + S], np.array(lab[s : s + S], np.int64)


def augment(seq, rng):
    if rng.random() < 0.5:
        seq = [np.ascontiguousarray(f[:, ::-1]) for f in seq]
    if rng.random() < 0.3:
        g = rng.uniform(0.6, 1.4)
        seq = [np.clip(f.astype(np.float32) * g, 0, 255).astype(np.uint8) for f in seq]
    if rng.random() < 0.5:
        q = [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(35, 85)]
        seq = [cv2.imdecode(cv2.imencode(".jpg", f, q)[1], cv2.IMREAD_COLOR) for f in seq]
    return seq


def batch(clips, rng, B, S=64, aug=True, same_source=0.0, junk=0.0, real=None, real_p=0.0, cs=None, cs_p=0.0,
          dirty_cut=0.0, dirty_same=0.0):
    """real: labeled windows from labels.load_windows(); each sequence is one with
    chance real_p (their unlabeled frames are -100, which the loss ignores).
    cs: ClipShots windows (clipshots.load_windows()), used with chance cs_p.
    Returns frames (B,S,3,H,W) and labels (B,S)."""
    xs, ys = [], []
    for _ in range(B):
        r = rng.random()
        if real and real_p > 0 and r < real_p:
            seq, y = labels.sample(real, rng, S)
        elif cs and cs_p > 0 and r < real_p + cs_p:
            if CS_HARDNEG and rng.random() < CS_HARDNEG_P:  # windows around a model's false alarms
                seq, y = clipshots.sample(CS_HARDNEG, rng, S)
            elif CS_FADENEG and rng.random() < CS_FADENEG_P:  # windows around a model's fade false alarms
                seq, y = clipshots.sample(CS_FADENEG, rng, S)
            elif CS_GRADUAL and rng.random() < CS_GRADUAL_P:  # a window holding a gradual transition
                seq, y = clipshots.sample(CS_GRADUAL, rng, S)
            else:
                seq, y = clipshots.sample(cs, rng, S)
        else:
            seq, y = make_sequence(clips, rng, S, same_source, junk, dirty_cut, dirty_same)
        if aug:
            seq = augment(seq, rng)
        xs.append(np.stack(seq))
        ys.append(y)
    x = torch.from_numpy(np.stack(xs).astype(np.float32) / 255).permute(0, 1, 4, 2, 3).contiguous()
    return x, torch.from_numpy(np.stack(ys))


def loss_fn(logits, y, w):
    """Class-weighted cross-entropy over the labelled frames (y >= 0); the rest are ignored."""
    logp = F.log_softmax(logits, -1)
    tw = F.one_hot(y.clamp(min=0), 3).float() * (y >= 0)[..., None] * w
    return -(tw * logp).sum() / tw.sum().clamp(min=1e-6)


# ---------------- train ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True,
                    help="directory of single-shot training clips, named <source video id>-<start frame>-<frames>.mp4")
    ap.add_argument("--val-clips", help="directory of single-shot clips from other source videos; with it, the "
                                        "progress line reports per-frame recall on synthetic sequences built from them")
    ap.add_argument("--videos", help="directory holding the labeled videos as <video_id>.mp4 (needed with --real "
                                     "until their frames are cached)")
    ap.add_argument("--device", help="cuda, mps or cpu (default: the first of those that is available)")
    ap.add_argument("--out", default="leanscene.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--spatial", action="store_true", help="add per-position frame-change features")
    ap.add_argument("--width", type=float, default=1.0, help="scale CNN channels")
    ap.add_argument("--dilations", default="1,2,4,8",
                    help="temporal block dilations; each frame sees 1 + 2*sum frames (1,2,4,8 -> 31; 1,2,4,8,16 -> 63)")
    ap.add_argument("--no-balance-sources", dest="balance_sources", action="store_false",
                    help="sample clips uniformly instead of source videos uniformly")
    ap.add_argument("--same-source", type=float, default=0.0, help="chance a shot follows one from the same source video")
    ap.add_argument("--junk", type=float, default=0.0, help="chance a hard cut is preceded by 1-2 unrelated frames")
    ap.add_argument("--dirty-cut", type=float, default=0.0,
                    help="chance a synthetic hard cut cuts off a disturbance (blur, leak, tint...) in the outgoing shot")
    ap.add_argument("--dirty-same", type=float, default=0.0,
                    help="chance per shot of the --dirty-cut disturbance inside the shot, recovering to the same shot (labeled same)")
    ap.add_argument("--same-source-near", type=int, default=0,
                    help="with --same-source: prefer a clip starting within this many frames of the previous one in the same video")
    ap.add_argument("--dirty-kinds", help="comma list of synth.DIRTY_KINDS to use (default all)")
    ap.add_argument("--dirty-same-whip", type=float, default=0.0,
                    help="share of --dirty-same negatives that are whip pans instead")
    ap.add_argument("--real", type=float, default=0.0,
                    help="share of sequences taken from windows around the hand labels (labels.py)")
    ap.add_argument("--clipshots", type=float, default=0.0,
                    help="share of sequences taken from ClipShots windows (clipshots.py)")
    ap.add_argument("--clipshots-hardneg", type=float, default=0.0,
                    help="share of the ClipShots sequences drawn from hardneg.py's windows")
    ap.add_argument("--clipshots-gradual", type=float, default=0.0,
                    help="share of the (non-hard-negative) ClipShots sequences drawn only from windows that "
                         "hold a gradual transition (they are ~20%% of the windows otherwise)")
    ap.add_argument("--clipshots-fadeneg", type=float, default=0.0,
                    help="share of the remaining ClipShots sequences drawn from DATA/clipshots_fadeneg (cleaned fade false alarms)")
    ap.add_argument("--hardneg-clean", action="store_true",
                    help="use the cleaned hard negatives (hardneg.py clean): likely unannotated transitions ignored")
    ap.add_argument("--init", help="start from this checkpoint's weights instead of random ones")
    ap.add_argument("--real-labels", help="a labels file in the format of labels/labels.csv, to read instead of it")
    ap.add_argument("--clip-list", help="train only on these clip filenames (one per line), to compare runs on identical data")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.state (saved at every eval)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="keep an exponential moving average of the weights with this decay (e.g. 0.999); "
                         "it is what gets evaluated and saved")
    args = ap.parse_args()

    global DIRTY_KINDS, DIRTY_SAME_WHIP, SAME_SOURCE_NEAR, CS_HARDNEG, CS_HARDNEG_P, CS_GRADUAL, CS_GRADUAL_P, CS_FADENEG, CS_FADENEG_P
    SAME_SOURCE_NEAR = args.same_source_near
    DIRTY_KINDS = args.dirty_kinds.split(",") if args.dirty_kinds else None
    DIRTY_SAME_WHIP = args.dirty_same_whip
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    print("loading clips...", flush=True)
    train_paths = synth.clip_paths(args.clips)
    if not train_paths:
        raise SystemExit(f"--clips: no .mp4 files in {args.clips}")
    if args.clip_list:
        keep = set(Path(args.clip_list).read_text().split())
        train_paths = [p for p in train_paths if p.name in keep]
    train_clips = [synth.read_clip(p, (100, 100)) for p in train_paths]
    if args.balance_sources:
        train_clips = ClipSet(train_clips, [synth.clip_source(p) for p in train_paths],
                              [int(p.stem.rsplit("-", 2)[1]) for p in train_paths])
        print(f"{len(train_clips)} train clips from {len(train_clips.groups)} source videos (sampled evenly)", flush=True)
    real = labels.load_windows(args.videos, labels_file=args.real_labels) if args.real > 0 else None
    if real is not None:
        if not real:
            raise SystemExit("--real: no labeled windows; pass --videos (see training/README.md)")
        print(f"{len(real)} labeled windows (used for {args.real:.0%} of sequences)", flush=True)
    cs = clipshots.load_windows() if args.clipshots > 0 else None
    if cs is not None and not len(cs):
        raise SystemExit(f"--clipshots: no windows in {clipshots.OUT}; run training/clipshots.py first")
    if cs is not None and args.clipshots_gradual > 0:
        CS_GRADUAL_P = args.clipshots_gradual
        CS_GRADUAL = cs.where(lambda t: (t == 2).any())
        print(f"[clipshots] {len(CS_GRADUAL)} windows hold a gradual transition", flush=True)
    if cs is not None and args.clipshots_fadeneg > 0:
        import hardneg
        CS_FADENEG_P = args.clipshots_fadeneg
        CS_FADENEG = hardneg.load_clean(src=clipshots.DATA / "clipshots_fadeneg")
        if not CS_FADENEG:
            raise SystemExit("--clipshots-fadeneg: no windows; run training/hardneg.py with HARDNEG_DIR=clipshots_fadeneg FADE_ONLY=1")
    if cs is not None and args.clipshots_hardneg > 0:
        import hardneg
        CS_HARDNEG_P = args.clipshots_hardneg
        CS_HARDNEG = hardneg.load_clean() if args.hardneg_clean else clipshots.load_windows(src=hardneg.OUT)
        if not len(CS_HARDNEG):
            raise SystemExit("--clipshots-hardneg: no windows; run training/hardneg.py first")
    xv = yv = None
    if args.val_clips:
        held_clips = [synth.read_clip(p, (100, 100)) for p in synth.clip_paths(args.val_clips)]
        xv, yv = batch(held_clips, random.Random(123), 96, aug=False)

    model = LeanScene(spatial=args.spatial, width=args.width,
                      dilations=[int(d) for d in args.dilations.split(",")]).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True)["model"])
        print(f"initialised from {args.init}", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    ema = (torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True)
        if args.ema > 0 else None)
    w = torch.tensor([1.0, 4.0, 2.0], device=device)
    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.out).with_suffix(".state")
    start, done_min = 1, 0.0
    if args.resume:
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if ema is not None:
            ema.load_state_dict(st["ema"])
        rng.setstate(st["rng"]); torch.set_rng_state(st["torch_rng"].cpu())
        start, done_min = st["step"] + 1, st["minutes"]
        print(f"resumed at step {start}", flush=True)
    t0 = t = time.time() - done_min * 60
    tot = 0.0
    for step in range(start, args.steps + 1):
        model.train()
        x, y = batch(train_clips, rng, args.bs, same_source=args.same_source, junk=args.junk,
                     real=real, real_p=args.real, cs=cs, cs_p=args.clipshots, dirty_cut=args.dirty_cut,
                     dirty_same=args.dirty_same)
        loss = loss_fn(model(x.to(device)), y.to(device), w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if ema is not None:
            ema.update_parameters(model)
        tot += loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            final = ema.module if ema is not None else model
            final.eval()
            line = f"step {step}: loss {tot/args.eval_every:.3f}"
            if xv is not None:
                with torch.no_grad():
                    pred = torch.cat([final(xv[i : i + 16].to(device)).argmax(2).cpu() for i in range(0, len(xv), 16)])
                acc = {k: float((pred[yv == k] == k).float().mean()) for k in range(3)}
                line += f"  val per-frame recall same {acc[0]:.3f} cut {acc[1]:.3f} fade {acc[2]:.3f}"
            print(f"{line}  ({time.time()-t:.0f}s)", flush=True)
            t, tot = time.time(), 0.0
            torch.save({"config": model.config, "model": final.state_dict()}, args.out)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None,
                        "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "step": step,
                        "minutes": (time.time() - t0) / 60}, state_path)
    print("saved", args.out)


if __name__ == "__main__":
    main()
