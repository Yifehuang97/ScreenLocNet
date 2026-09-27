"""Architectures for the gaze-only baselines.

The three permutation-invariant variants are exactly invariant to frame order; the
temporal variant adds sinusoidal positional encoding so that order can matter.
"""
import os
import math
import torch
import torch.nn as nn
RESULTS = os.environ.get("SCRLOC_RESULTS", "baselines/results")
os.makedirs(RESULTS, exist_ok=True)
import torch.nn.functional as F


def two_corners_sc(pred):
    """Paper's 8-DOF geometric parameterisation -> 4 corners (B,12)."""
    tl, br = pred[:, :3], pred[:, 3:6]
    center = (tl + br) / 2
    theta = torch.sigmoid(pred[:, 6]) * (2 * math.pi)
    phi = torch.sigmoid(pred[:, 7]) * math.pi
    r = torch.norm(tl - center, dim=1)
    bl = torch.stack([r * torch.sin(phi) * torch.cos(theta),
                      r * torch.sin(phi) * torch.sin(theta),
                      r * torch.cos(phi)], dim=1) + center
    tr = 2 * center - bl
    return torch.cat([tl, tr, bl, br], dim=1)


def sinusoid(seq_len, dim, device):
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    idx = torch.arange(dim, device=device).unsqueeze(0).float()
    ang = pos / (10000 ** (2 * (idx // 2) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(ang[:, 0::2])
    pe[:, 1::2] = torch.cos(ang[:, 1::2])
    return pe.unsqueeze(0)


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 8))

    def forward(self, x):
        return two_corners_sc(self.net(x))


class MAB(nn.Module):
    """Multihead attention block (Set Transformer, Lee et al. 2019)."""
    def __init__(self, dim, nhead):
        super().__init__()
        self.att = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, q, k):
        h = self.ln0(q + self.att(q, k, k, need_weights=False)[0])
        return self.ln1(h + self.ff(h))


class ISAB(nn.Module):
    """Induced set attention block: O(T*m) and exactly permutation-equivariant."""
    def __init__(self, dim, nhead, m=32):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, dim) * 0.02)
        self.mab0, self.mab1 = MAB(dim, nhead), MAB(dim, nhead)

    def forward(self, x):
        h = self.mab0(self.I.expand(x.shape[0], -1, -1), x)
        return self.mab1(x, h)


class PMA(nn.Module):
    """Pooling by multihead attention -> permutation-invariant readout."""
    def __init__(self, dim, nhead, k=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.mab = MAB(dim, nhead)

    def forward(self, x):
        return self.mab(self.S.expand(x.shape[0], -1, -1), x).mean(1)


class GazeBaseline(nn.Module):
    def __init__(self, kind, dim=256, nlayer=4, nhead=8, n_sub=120, nbins=32):
        super().__init__()
        self.kind, self.dim, self.n_sub, self.nbins = kind, dim, n_sub, nbins

        if kind == "temporal":
            # multi-scale local windows, exactly as ScreenLocNet's dense branch
            self.p1 = nn.Conv1d(6, dim, kernel_size=20, stride=10)
            self.p2 = nn.Conv1d(6, dim, kernel_size=30, stride=15)
            self.p3 = nn.Conv1d(6, dim, kernel_size=40, stride=20)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, nhead, batch_first=True), nlayer)
            self.convs = nn.ModuleList([nn.Conv1d(dim, dim, 5, padding=2),
                                        nn.Conv1d(dim, dim, 3, padding=1)])
            self.head = Head(dim)

        elif kind == "settx":
            # Set Transformer (ISAB x nlayer + PMA) over all 1200 gaze rays.
            self.emb = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
            self.isabs = nn.ModuleList([ISAB(dim, nhead, m=32) for _ in range(nlayer)])
            self.pma = PMA(dim, nhead)
            self.head = Head(dim)

        elif kind == "deepsets":
            w = 2 * dim  # widened so total capacity matches the other variants
            self.phi = nn.Sequential(nn.Linear(6, w), nn.GELU(),
                                     nn.Linear(w, w), nn.GELU(),
                                     nn.Linear(w, w), nn.GELU(),
                                     nn.Linear(w, dim), nn.GELU())
            self.rho = nn.Sequential(nn.Linear(2 * dim, w), nn.GELU(),
                                     nn.Linear(w, w), nn.GELU(),
                                     nn.Linear(w, dim), nn.GELU())
            self.head = Head(dim)

        elif kind == "density":
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.GELU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.GELU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1), nn.GELU(),
                nn.Conv2d(512, 512, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1))
            self.proj = nn.Sequential(nn.Linear(512, 2 * dim), nn.GELU(),
                                      nn.Linear(2 * dim, dim), nn.GELU())
            self.head = Head(dim)
        else:
            raise ValueError(kind)

    def _sub(self, x):
        """Uniformly subsample frames (keeps memory sane for the token models)."""
        T = x.shape[1]
        idx = torch.linspace(0, T - 1, self.n_sub, device=x.device).long()
        return x[:, idx]

    def forward(self, face, gaze):
        x = torch.cat([face, gaze], dim=2)              # (B,T,6)

        if self.kind == "temporal":
            xp = x.permute(0, 2, 1)
            t = torch.cat([self.p1(xp), self.p2(xp), self.p3(xp)], dim=2).permute(0, 2, 1)
            t = t + sinusoid(t.shape[1], self.dim, t.device)   # PE ON: order matters
            t = self.enc(t)
            t = t.permute(0, 2, 1)
            for c in self.convs:
                t = F.gelu(c(t))
            return self.head(t.permute(0, 2, 1).mean(1))

        if self.kind == "settx":
            t = self.emb(x)                             # no PE -> permutation invariant
            for blk in self.isabs:
                t = blk(t)
            return self.head(self.pma(t))

        if self.kind == "deepsets":
            h = self.phi(x)
            return self.head(self.rho(torch.cat([h.mean(1), h.max(1).values], dim=1)))

        # density: 2D histogram of gaze direction (azimuth, elevation)
        g = gaze / (gaze.norm(dim=2, keepdim=True) + 1e-8)
        az = torch.atan2(g[..., 0], g[..., 2].clamp(min=1e-6))
        el = torch.asin(g[..., 1].clamp(-1 + 1e-6, 1 - 1e-6))
        B, nb = g.shape[0], self.nbins
        ia = ((az + math.pi / 2) / math.pi * nb).clamp(0, nb - 1).long()
        ie = ((el + math.pi / 2) / math.pi * nb).clamp(0, nb - 1).long()
        flat = (ie * nb + ia)
        hist = torch.zeros(B, nb * nb, device=g.device)
        hist.scatter_add_(1, flat, torch.ones_like(flat, dtype=hist.dtype))
        hist = (hist / hist.sum(1, keepdim=True).clamp(min=1)).view(B, 1, nb, nb)
        f = self.cnn(hist).flatten(1)
        return self.head(self.proj(f))
