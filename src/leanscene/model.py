"""The LeanScene network: a per-frame encoder and a dilated 1D temporal head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_SIZE = 100  # frames are resized to FRAME_SIZE x FRAME_SIZE before the network sees them
HIST_BINS = 4  # per channel; the joint colour histogram has HIST_BINS ** 3 = 64 values


def _conv_bn(i: int, o: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class _TBlock(nn.Module):
    """Residual block of one dilated 1D convolution."""

    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return x + F.relu(self.bn(self.conv(x)))


def color_hist(x: torch.Tensor, bins: int = HIST_BINS) -> torch.Tensor:
    """x: (N,3,H,W) in [0,1] -> (N, bins**3) normalised joint colour histogram."""
    q = (x * (bins - 0.001)).long()
    idx = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    return F.one_hot(idx.flatten(1), bins**3).float().mean(1)


class LeanScene(nn.Module):
    """Labels every frame of a video as same (0), cut (1) or fade (2).

    Each frame is encoded once: a four-layer 2D CNN gives a global feature and a 2x2 grid of
    features, and two descriptors come straight from the pixels (a 64-bin colour histogram
    and an 8x8 colour thumbnail). The temporal head sees those features together with their
    change since the previous frame, and runs residual dilated 1D convolutions over the
    sequence.

    A cut is labeled on the first frame of the new shot; fade marks the frames inside a
    gradual transition.
    """

    def __init__(self, feat: int = 128, hid: int = 128, dilations=(1, 2, 4, 8, 16), spatial: bool = True,
                 width: float = 1.0):
        super().__init__()
        self.config = {"feat": feat, "hid": hid, "dilations": list(dilations), "spatial": spatial, "width": width}
        c = [max(8, int(round(ch * width))) for ch in (24, 48, 96)]
        self.enc = nn.Sequential(_conv_bn(3, c[0], 2), _conv_bn(c[0], c[1], 2), _conv_bn(c[1], c[2], 2),
                                 _conv_bn(c[2], feat, 2))
        self.feat, self.spatial = feat, spatial
        self.sdim = 3 * 8 * 8 + feat * 2 * 2 if spatial else 0
        extra = 0
        if spatial:
            self.sproj = nn.Sequential(nn.Conv1d(self.sdim, 64, 1), nn.ReLU(inplace=True))
            extra = 64
        self.inp = nn.Conv1d(feat + 64 + 1 + extra, hid, 1)
        self.blocks = nn.Sequential(*[_TBlock(hid, d) for d in dilations])
        self.out = nn.Conv1d(hid, 3, 1)

    @property
    def receptive_field(self) -> int:
        """Number of frames that can influence one frame's label."""
        return 1 + 2 * sum(self.config["dilations"])

    def frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3,H,W) float in [0,1], BGR channel order -> (N, D) per-frame features."""
        fmap = self.enc(x)
        parts = [fmap.mean((2, 3)), color_hist(x)]
        if self.spatial:
            # Fixed-kernel pooling on a centre crop of divisible size, so it also runs on
            # backends without adaptive pooling for non-divisible sizes.
            h, w = x.shape[2] // 8 * 8, x.shape[3] // 8 * 8
            top, left = (x.shape[2] - h) // 2, (x.shape[3] - w) // 2
            xc = x[:, :, top:top + h, left:left + w]
            fh, fw = fmap.shape[2] // 2 * 2, fmap.shape[3] // 2 * 2
            parts += [F.avg_pool2d(xc, (h // 8, w // 8)).flatten(1),
                      F.avg_pool2d(fmap[:, :, :fh, :fw], (fh // 2, fw // 2)).flatten(1)]
        return torch.cat(parts, 1)

    def head(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B,S,D) frame features in temporal order -> (B,S,3) logits."""
        f = f.transpose(1, 2)  # (B, D, S)
        k = self.feat + 64
        hist = f[:, self.feat:k]
        diff = torch.cat([torch.zeros_like(hist[:, :, :1]), (hist[:, :, 1:] - hist[:, :, :-1]).abs()], 2)
        x = [f[:, :k], diff.sum(1, keepdim=True)]
        if self.spatial:
            sp = f[:, k:]
            sdiff = torch.cat([torch.zeros_like(sp[:, :, :1]), (sp[:, :, 1:] - sp[:, :, :-1]).abs()], 2)
            x.append(self.sproj(sdiff))
        return self.out(self.blocks(self.inp(torch.cat(x, 1)))).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,S,3,H,W) float in [0,1], BGR -> (B,S,3) logits."""
        b, s = x.shape[:2]
        return self.head(self.frame_features(x.flatten(0, 1)).reshape(b, s, -1))
