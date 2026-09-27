"""Evaluation metrics, in the exact form used for the reported numbers."""
import numpy as np

# distance from the screen centre to a typical viewer, averaged over the dataset
VIEWER_DEPTH = 0.41873748921235804


def corner_dist(pred, gt):
    """Mean L2 over the four corners, per clip, in centimetres. pred/gt are (N, 4, 3) metres."""
    return np.sqrt(((pred - gt) ** 2).sum(-1)).mean(-1) * 100.0


def center_dist(pred, gt):
    return np.sqrt(((pred.mean(1) - gt.mean(1)) ** 2).sum(-1)) * 100.0


def _iou_2d(a, b):
    from shapely.geometry import Polygon
    p, q = Polygon(a), Polygon(b)
    if not p.is_valid:
        p = p.buffer(0)
    if not q.is_valid:
        q = q.buffer(0)
    if p.area == 0 or q.area == 0:
        return None
    u = p.union(q).area
    return p.intersection(q).area / u if u > 0 else None


def vp_iou(pred, gt, depth=VIEWER_DEPTH):
    """Viewer-Perspective IoU: overlap seen from a viewer `depth` in front of the screen."""
    out = []
    for p, g in zip(pred, gt):
        tl, tr, bl, br = g
        normal = np.cross(tr - tl, br - tl)
        n = np.linalg.norm(normal)
        if n == 0:
            out.append(0.0)
            continue
        normal = normal / n
        centre = g.mean(0)
        eye = centre + depth * normal
        plane = centre + 2 * depth * normal
        b1 = (tr - tl) / np.linalg.norm(tr - tl)
        b2 = (br - tl) / np.linalg.norm(br - tl)

        def to2d(pt):
            v = pt - eye
            t = np.dot(plane - eye, normal) / np.dot(v, normal)
            q = eye + t * v - plane
            return [np.dot(q, b1), np.dot(q, b2)]

        try:
            iou = _iou_2d(np.array([to2d(x) for x in (g[0], g[1], g[3], g[2])]),
                          np.array([to2d(x) for x in (p[0], p[1], p[3], p[2])]))
        except Exception:
            iou = None
        out.append(0.0 if iou is None else max(iou, 0.0))
    return np.array(out)


def recording_bootstrap(per_clip, recording_ids, n=5000, seed=0):
    """95% CI that resamples whole recordings.

    Clips overlap by up to 600 frames, so a clip-level bootstrap understates the
    interval by roughly a factor of four. Always report this one.
    """
    rng = np.random.default_rng(seed)
    per_clip = np.asarray(per_clip)
    recs = np.unique(recording_ids)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(recs), len(recs))
        draws.append(np.concatenate([per_clip[recording_ids == recs[p]] for p in pick]).mean())
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize(pred, gt, recording_ids=None):
    cd, ce = corner_dist(pred, gt), center_dist(pred, gt)
    out = {
        "Corner-Dist": float(cd.mean()),
        "Center-Dist": float(ce.mean()),
        "Center-Dist median": float(np.median(ce)),
        "Center-Dist p90": float(np.percentile(ce, 90)),
        "VP-IoU": float(vp_iou(pred, gt).mean()),
    }
    if recording_ids is not None:
        lo, hi = recording_bootstrap(ce, np.asarray(recording_ids))
        out["Center-Dist 95% CI (recording-level)"] = [round(lo, 2), round(hi, 2)]
    return out
