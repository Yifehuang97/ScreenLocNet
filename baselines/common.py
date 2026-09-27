"""Shared data loading and metrics for the baseline comparisons."""
import os
import os, json
import numpy as np

RESULTS = os.environ.get("SCRLOC_RESULTS", "baselines/results")
os.makedirs(RESULTS, exist_ok=True)
import os
# point SCRLOC_DATA at the directory created by scripts/download_data.py
ROOT = os.path.join(os.environ.get("SCRLOC_DATA", "data/ScrLoc30Exp"), "bg_ssl")
NPZ = os.path.join(ROOT, "npz", "{:05d}.npz")
SPLIT = os.path.join(ROOT, "final_splits", "{}_split.json")
CAMPOS = os.path.join(ROOT, "final_splits", "id_campos_map.json")


def load_split(split_type, split):
    with open(SPLIT.format(split_type)) as f:
        return json.load(f)[split]


def load_clips(ids, verbose=False):
    """Return dict of stacked arrays for the given clip ids (order preserved)."""
    gaze, corners, rid, sid = [], [], [], []
    for i, cid in enumerate(ids):
        d = np.load(NPZ.format(cid))
        gaze.append(d["gaze_dir"].astype(np.float32))
        corners.append(np.stack([d["tl"], d["tr"], d["bl"], d["br"]]).astype(np.float32))
        rid.append(int(str(d["id"])))
        sid.append(int(str(d["session_id"])))
        if verbose and i % 500 == 0:
            print(f"  loaded {i}/{len(ids)}", flush=True)
    return {
        "gaze": np.stack(gaze),            # (N, 1200, 3)
        "corners": np.stack(corners),      # (N, 4, 3)
        "rid": np.array(rid),              # recording id -> camera position
        "sid": np.array(sid),
        "cid": np.array(ids),
    }


def load_campos():
    with open(CAMPOS) as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


# ---------------- metrics (identical definitions to the main paper) -----------
def corner_dist(pred, gt):
    """pred,gt: (N,4,3) in metres -> per-clip mean L2 over the 4 corners, in cm."""
    return np.sqrt(((pred - gt) ** 2).sum(-1)).mean(-1) * 100.0


def center_dist(pred, gt):
    return np.sqrt(((pred.mean(1) - gt.mean(1)) ** 2).sum(-1)) * 100.0


# ---- VP-IoU comes from the package so there is exactly one implementation -------
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from screenlocnet.metrics import vp_iou  # noqa: E402


def evaluate(pred, gt):
    cd, ce = corner_dist(pred, gt), center_dist(pred, gt)
    return {
        "Center-Dist": float(ce.mean()),
        "Corner-Dist": float(cd.mean()),
        "VP-IoU": float(vp_iou(pred, gt).mean()),
        "_corner_per_clip": cd,
        "_center_per_clip": ce,
    }
