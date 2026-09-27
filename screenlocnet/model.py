"""ScreenLocNet.

Two branches feed one prediction head.

  scene-static   DUSt3R face point maps and DINOv2 face tokens, 32 sampled frames.
                 This is what carries metric scale, since the point map is anchored
                 by the face.
  gaze-dynamic   the 1,200-frame gaze-ray sequence, aggregated by three Conv1d
                 windows of different lengths.

A note on what this architecture is, because the naming in the paper was loose.
No positional encoding is ever added to the transformer inputs and the pooling is a
mean, so both encoders are permutation-equivariant. Frame order therefore enters only
through the local Conv1d windows. Shuffling the frames costs 11.22 -> 11.99 cm, which
is what this design predicts. The branch is best understood as a multi-scale *set*
encoder over gaze rays rather than a temporal model.
"""
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .geometry import DOF, decode


class PointMapTokenizer(nn.Module):
    """One token per frame from that frame's DUSt3R face point map."""

    def __init__(self, dim: int = 768):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, dim)

    def forward(self, x):                      # (B, T, 3, H, W)
        b, t = x.shape[:2]
        f = self.backbone(rearrange(x, "b t c h w -> (b t) c h w")).flatten(1)
        return rearrange(self.proj(f), "(b t) d -> b t d", b=b, t=t)


class Projector(nn.Module):
    """Pre-norm Conv1d projection with a residual feed-forward."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, out_dim, 3, padding=1)
        self.ff = nn.Conv1d(out_dim, out_dim, 3, padding=1)
        self.in_norm = nn.LayerNorm(in_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x):                      # (B, C, T)
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(self.proj(x))
        y = self.out_norm(x.transpose(1, 2)).transpose(1, 2)
        return x + self.ff(y)


class PredictionHead(nn.Module):
    def __init__(self, dim: int, pred_type: str = "two_corners_sc", hidden=None):
        super().__init__()
        hidden = hidden or [dim // 2, dim // 4]
        sizes = [dim] + list(hidden) + [DOF[pred_type]]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.pred_type = pred_type

    def forward(self, x):
        return decode(self.pred_type, self.mlp(x))


class ScreenLocNet(nn.Module):
    """Inputs (see `screenlocnet.data` for how they are produced):

        gaze_origin  (B, 1200, 3)            per-frame head position
        gaze_dir     (B, 1200, 3)            per-frame gaze direction from L2CS
        pointmap     (B, 32, 256, 256, 3)    DUSt3R face point maps
        dino_face    (B, 32, 768)            DINOv2 CLS tokens of the face crops

    Returns the four corners as (B, 12) in tl, tr, bl, br order, metres, in the
    webcam frame.
    """

    def __init__(self, dim: int = 768, num_layers: int = 8, num_heads: int = 8,
                 pred_type: str = "two_corners_sc"):
        super().__init__()
        self.pointmap_tokenizer = PointMapTokenizer(dim)
        self.dino_face_proj = Projector(768, dim)

        # three window lengths over the raw (origin, direction) sequence
        self.window_20 = nn.Conv1d(6, dim, kernel_size=20, stride=10)
        self.window_30 = nn.Conv1d(6, dim, kernel_size=30, stride=15)
        self.window_40 = nn.Conv1d(6, dim, kernel_size=40, stride=20)

        enc = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads), num_layers=num_layers
        )
        self.cross_modal_encoder = enc()
        self.multi_scale_encoder = enc()

        self.cross_modal_convs = nn.ModuleList(
            [nn.Conv1d(dim, dim, 5, padding=2), nn.Conv1d(dim, dim, 5, padding=2),
             nn.Conv1d(dim, dim, 3, padding=1)])
        self.multi_scale_convs = nn.ModuleList(
            [nn.Conv1d(dim, dim, 5, padding=2), nn.Conv1d(dim, dim, 5, padding=2),
             nn.Conv1d(dim, dim, 3, padding=1), nn.Conv1d(dim, dim, 3, padding=1)])

        self.head = PredictionHead(2 * dim, pred_type)

    @staticmethod
    def _apply_convs(tokens, convs):
        x = tokens.transpose(1, 2)
        for c in convs:
            x = F.gelu(c(x))
        return x.transpose(1, 2)

    def forward(self, gaze_origin, gaze_dir, pointmap, dino_face):
        rays = torch.cat([gaze_origin, gaze_dir], dim=2).transpose(1, 2)   # (B, 6, T)
        w20 = self.window_20(rays).transpose(1, 2)
        w30 = self.window_30(rays).transpose(1, 2)
        w40 = self.window_40(rays).transpose(1, 2)

        pm = self.pointmap_tokenizer(pointmap.permute(0, 1, 4, 2, 3))
        df = self.dino_face_proj(dino_face.transpose(1, 2)).transpose(1, 2)

        # No positional encoding is added here, by design; see the module docstring.
        cross = self.cross_modal_encoder(torch.cat([df, w20, pm], dim=1))
        multi = self.multi_scale_encoder(torch.cat([w20, w30, w40], dim=1))

        cross = self._apply_convs(cross, self.cross_modal_convs).mean(dim=1)
        multi = self._apply_convs(multi, self.multi_scale_convs).mean(dim=1)
        return self.head(torch.cat([cross, multi], dim=1))
