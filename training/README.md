# Training

The code that trained the bundled weights and every model in the paper: the trainer, the
synthetic transitions, the ClipShots windows, and the mining and cleaning of hard negatives
that the paper's data ablation is about.

Run everything from the repository root, with the package installed (`pip install -e .`).

| File | What it does |
|---|---|
| `train.py` | The trainer. Mixes three kinds of 64-frame sequences: synthetic, hand-labeled, ClipShots |
| `synth.py` | Cuts, dissolves, dips, and the same-shot disturbances used as negatives (flashes, exposure swings, whip pans, lens flares, occlusions, "dirty" cuts) |
| `labels.py` | Windows around the hand labels in [`labels/`](../labels/README.md) |
| `clipshots.py` | ClipShots training videos as 64-frame windows; also extracts the validation and test videos |
| `hardneg.py` | Mines windows around a model's false alarms on ClipShots training videos, and cleans them |
| `clipshots_val_videos.txt` | The 906 ClipShots *train* videos held out as the validation set (thresholds and data steps were chosen on them; they are never trained on or mined) |

## What you need to bring

| Data | Used for | Where it comes from |
|---|---|---|
| Single-shot clips | 45% of training sequences (synthetic) | **Not distributed.** Ours are 397 hand-checked single-shot excerpts from 37 skateboarding videos. Any directory of single-shot clips works: `.mp4` files named `<source video id>-<start frame>-<frames>.mp4`, at least one of them 64 frames or longer (clips shorter than 8 frames are never used). The source id is used to sample source videos evenly |
| The labeled videos | 25% (`--real`) | The 466 YouTube videos listed in [`labels/videos.csv`](../labels/videos.csv), downloaded as `<video_id>.mp4` into one directory (command in [`labels/README.md`](../labels/README.md)) |
| ClipShots | 30% (`--clipshots`) | The three archive parts `ClipShots-a`, `-b`, `-c` (48 GB) and the annotations, both linked from [github.com/Tangshitao/ClipShots](https://github.com/Tangshitao/ClipShots). Put a copy of that repository at `training/data/ClipShots` |

Because the clips are not distributed, the bundled weights cannot be retrained exactly. With
your own clips the same code and commands train the same network on the same real data.

## The paper's recipe

Each step is one command. `$PARTS` is the directory holding the three ClipShots archive parts.

```sh
# 1. ClipShots windows (about 20 minutes from local parts), and the validation videos
python training/clipshots.py $PARTS
python training/clipshots.py --val $PARTS

BASE="--clips clips/ --videos videos/ --steps 12000 --eval_every 2000 --spatial \
  --dilations 1,2,4,8,16 --dirty-cut 0.3 --dirty-same 0.12 --dirty-kinds leak,white,tint,dark,occluder \
  --clipshots 0.3 --real 0.25"

# 2. Base model at the development width (LeanScene-2x in the paper)
python training/train.py $BASE --width 1 --seed 1 --out training/data/runs/base.pt

# 3. Mine its false alarms on ClipShots training videos, then clean them
MAX_VIDEOS=4000 python training/hardneg.py $PARTS training/data/runs/base.pt
python training/hardneg.py clean training/data/runs/base.pt

# 4. Retrain with the cleaned negatives and more gradual transitions
python training/train.py $BASE --width 1 --seed 0 --out training/data/runs/gradual.pt \
  --clipshots-hardneg 0.25 --hardneg-clean --clipshots-gradual 0.4

# 5. Mine and clean that model's fade false alarms
HARDNEG_DIR=clipshots_fadeneg FADE_ONLY=1 WORKERS=1 MAX_VIDEOS=1468 python training/hardneg.py $PARTS training/data/runs/gradual.pt
HARDNEG_DIR=clipshots_fadeneg python training/hardneg.py clean training/data/runs/gradual.pt

# 6. The reported model: the full recipe at half that width, seeds 0 to 4
python training/train.py $BASE --width 0.5 --seed 2 --out training/data/runs/leanscene_seed2.pt \
  --clipshots-hardneg 0.25 --hardneg-clean --clipshots-fadeneg 0.2 --clipshots-gradual 0.4
```

Step 6 with `--seed 2` is the run behind the bundled weights. It took about 66 minutes on an Apple
M1 Pro (MPS). Score the result with [`eval/`](../eval/README.md): `--model training/data/runs/leanscene_seed2.pt`.

The rows of the paper's data ablation are the same command with flags removed: no
`--clipshots-*` flags (base recipe), `--clipshots-hardneg 0.4` without `--hardneg-clean`
(mined, as annotated), `--clipshots-hardneg 0.25 --hardneg-clean` (cleaned), plus
`--clipshots-gradual 0.4` (oversampling), plus `--clipshots-fadeneg 0.2` (full recipe), and the
full recipe without `--real` (no in-domain labels). Those rows were trained with `--width 1`.

Two details of our runs that the commands above do not reproduce: the model that mined in
step 3 had been trained on an earlier, smaller set of ClipShots windows, and the fade mining in
step 5 stopped at 1,468 videos because the machine ran out of disk, not by design.

## `train.py` flags

| Flag | Default | Meaning |
|---|---|---|
| `--clips DIR` | required | Single-shot training clips |
| `--val-clips DIR` | none | Clips from other source videos; adds per-frame recall on synthetic sequences to the progress line. Not used to choose anything |
| `--videos DIR` | none | The labeled videos; needed with `--real` until their frames are cached |
| `--out PATH` | `leanscene.pt` | Checkpoint, loadable with `leanscene.load_model(PATH)` and `leanscene --model PATH` |
| `--steps N`, `--bs N`, `--lr X`, `--seed N` | 3000, 12, 2e-3, 0 | AdamW (weight decay 1e-4), one-cycle schedule |
| `--eval_every N` | 250 | Steps between progress lines; the checkpoint and resume state are saved at each |
| `--resume` | off | Continue from `<out>.state` |
| `--device D` | auto | `cuda`, `mps` or `cpu` |
| `--spatial` | off | The layout features (8x8 thumbnail, 2x2 feature grid). On in every reported model |
| `--width X` | 1.0 | Encoder width. The reported model is 0.5; 1.0 is LeanScene-2x |
| `--dilations LIST` | `1,2,4,8` | Temporal dilations. The paper uses `1,2,4,8,16` (63-frame receptive field) |
| `--real P` | 0 | Share of sequences from the hand labels |
| `--real-labels FILE` | `labels/labels.csv` | Another labels file in the same format |
| `--clipshots P` | 0 | Share of sequences from ClipShots windows |
| `--clipshots-hardneg P` | 0 | Share of the ClipShots sequences drawn from mined false alarms |
| `--hardneg-clean` | off | Use the cleaned targets (`clean.npz`) for those |
| `--clipshots-fadeneg P` | 0 | Share of the remaining ClipShots sequences drawn from cleaned fade false alarms |
| `--clipshots-gradual P` | 0 | Share of the remaining ones drawn only from windows holding a gradual transition |
| `--dirty-cut P`, `--dirty-same P`, `--dirty-kinds LIST` | 0, 0, all | Cuts that cut off a disturbance, the matching same-shot negative, and which disturbances |
| `--same-source P`, `--same-source-near N`, `--junk P`, `--dirty-same-whip P`, `--no-balance-sources`, `--clip-list FILE`, `--init PATH`, `--ema X` | off | Options from earlier experiments; not used by the paper's recipe |

`hardneg.py` is configured by environment variables: `MAX_VIDEOS` (default 1200), `WORKERS`
(default 3), `HARDNEG_DIR` (default `clipshots_hardneg`), `FADE_ONLY=1`, `LEANSCENE_DEVICE`.

## Data these scripts write

Everything goes under `training/data/` (gitignored), or under `$LEANSCENE_DATA` if that is set.
Sizes are what our runs measured in November 2026.

| Path | Written by | Read by | Size | Resume state | Safe to delete? |
|---|---|---|---|---|---|
| `ClipShots/` | You (a copy of the ClipShots repository) | `clipshots.py`, `hardneg.py` (annotations) | 4 MB | | Yes; copy it again |
| `clipshots_windows/` | `clipshots.py $PARTS` | `train.py --clipshots` | 4.2 GB: 19,506 windows from 3,749 videos, one `.pkl` per video | Videos with a `.pkl` are skipped | Yes; 20 minutes to rebuild from the parts |
| `clipshots_test/` | `clipshots.py $PARTS` | `eval/evaluate.py clipshots` | 2.1 GB: the 500 test videos | Existing files are skipped | Yes; same command |
| `clipshots_val/` | `clipshots.py --val $PARTS` | Choosing thresholds | 10 GB: the 906 validation videos | Existing files are skipped | Yes; 15 minutes to rebuild |
| `clipshots_hardneg/` | `hardneg.py $PARTS CKPT`, then `hardneg.py clean CKPT` (`clean.npz`, 1 MB) | `train.py --clipshots-hardneg` | 1.2 GB: 5,528 windows from 2,751 videos scanned | `done.jsonl` lists scanned videos | Yes; 3 h 40 min to mine again, minutes to clean |
| `clipshots_fadeneg/` | The same with `HARDNEG_DIR=clipshots_fadeneg FADE_ONLY=1` | `train.py --clipshots-fadeneg` | 0.7 GB: 2,934 windows from 1,468 videos | `done.jsonl` | Yes; 70 minutes to mine again |
| `label_frames/` | `labels.py`, `train.py --real` | `train.py --real` | One `.npz` per label, 81 frames at 100x100; not measured in this layout | Cached windows are not decoded again | Yes; the videos are decoded again |
| a temporary directory | `clipshots.py`, `hardneg.py` | themselves | A few videos at a time | | Removed at the end of a run; delete it after a crash |

Checkpoints go wherever `--out` points, with the resume state beside them as `<out>.state`
(the commands above use `training/data/runs/`; a few MB per run).

## What was and was not checked

This directory is the research code with its paths and imports changed to stand alone. On
2026-11-18, against the data and outputs of the original runs:

- `train.py` ran 20 steps of the step-6 command and loaded what the paper describes: 397 clips
  from 37 videos, 19,506 ClipShots windows of which 3,861 hold a gradual transition, 2,903
  cleaned hard negatives, 2,339 cleaned fade negatives, a 0.39 M-parameter model. The checkpoint
  loads with `leanscene.load_model`.
- `hardneg.py clean` reproduced the original `clean.npz` exactly on a sample of 152 windows.
- The code that builds synthetic sequences is unchanged from the original, byte for byte.
- `labels/labels.csv` holds the same 1,245 labels, in the same order per video, as the snapshot
  the runs read; `clipshots_val_videos.txt` matches the validation ground truth.

Not run in this layout: a full 12,000-step training, `clipshots.py` and the mining pass of
`hardneg.py` on the archive (their logic is unchanged; only paths differ), `--resume`, and
`labels.py` decoding the YouTube videos (the original runs read frames through a different
video reader; the new one was checked on local clips only).
