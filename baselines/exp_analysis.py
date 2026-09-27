"""Distributional statistics for a set of per-clip errors.

Reports medians and percentiles, and a bootstrap that resamples whole recordings
rather than clips. Clips from one recording overlap by up to 600 frames, so a
clip-level interval is roughly four times too narrow.
"""
import os
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("SCRLOC_RESULTS", "baselines/results")
os.makedirs(RESULTS, exist_ok=True)
import numpy as np, common

rng = np.random.default_rng(0)
campos = common.load_campos()
out = {}

for scene, split_type, path, scale in [
        ("Scene 2", "scene_1", "results/arrays/final_error_scene2.npy", 100.0),
        ("Scene 3", "scene_2", "results/arrays/final_error_scene3.npy", 1.0)]:
    err = np.load(path) * scale                      # cm
    ids = common.load_split(split_type, 'test')
    assert len(err) == len(ids), (len(err), len(ids))
    d = common.load_clips(ids)
    rid = d['rid']

    print(f"\n===== {scene}  (n={len(err)} clips, {len(set(rid))} recordings) =====")
    print(f"  mean {err.mean():.2f}  median {np.median(err):.2f}  "
          f"p25 {np.percentile(err,25):.2f}  p75 {np.percentile(err,75):.2f}  "
          f"p90 {np.percentile(err,90):.2f}  p95 {np.percentile(err,95):.2f}  max {err.max():.2f}")

    # ---- clip-level bootstrap (over-optimistic: clips overlap by up to 600 frames)
    bs = [err[rng.integers(0, len(err), len(err))].mean() for _ in range(5000)]
    clip_ci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))

    # ---- recording-level: cluster bootstrap, the statistically correct unit
    recs = np.unique(rid)
    per_rec = np.array([err[rid == r].mean() for r in recs])
    bsr = []
    for _ in range(5000):
        pick = rng.integers(0, len(recs), len(recs))
        bsr.append(np.concatenate([err[rid == recs[p]] for p in pick]).mean())
    rec_ci = (np.percentile(bsr, 2.5), np.percentile(bsr, 97.5))

    print(f"  clip-level 95% CI      [{clip_ci[0]:.2f}, {clip_ci[1]:.2f}]  (+/-{(clip_ci[1]-clip_ci[0])/2:.2f})")
    print(f"  recording-level 95% CI [{rec_ci[0]:.2f}, {rec_ci[1]:.2f}]  (+/-{(rec_ci[1]-rec_ci[0])/2:.2f})")
    print(f"  per-recording mean err: min {per_rec.min():.2f} median {np.median(per_rec):.2f} max {per_rec.max():.2f}")

    # ---- camera-position breakdown
    print("  by camera position:")
    cp = np.array([campos.get(int(r), -1) for r in rid])
    rows = []
    for c in sorted(set(cp[cp > 0])):
        m = cp == c
        rows.append((c, int(m.sum()), err[m].mean(), np.median(err[m])))
        print(f"    cam {c}: n={m.sum():4d}  mean {err[m].mean():5.2f}  median {np.median(err[m]):5.2f}")

    # ---- high-error tail
    hi = err > 30
    print(f"  clips >30cm: {hi.sum()}/{len(err)} ({100*hi.mean():.1f}%), "
          f"from {len(set(rid[hi]))} recording(s): {sorted(set(rid[hi].tolist()))}")

    out[scene] = dict(mean=float(err.mean()), median=float(np.median(err)),
                      p90=float(np.percentile(err, 90)), p95=float(np.percentile(err, 95)),
                      clip_ci=[float(x) for x in clip_ci], rec_ci=[float(x) for x in rec_ci],
                      n_clips=int(len(err)), n_rec=int(len(recs)),
                      campos=[[int(a), int(b), float(c), float(dd)] for a, b, c, dd in rows],
                      frac_gt30=float(hi.mean()))

json.dump(out, open(os.path.join(RESULTS, 'analysis_reported.json'), 'w'), indent=1)
print("\nsaved ->", os.path.join(RESULTS, "analysis_reported.json"))
