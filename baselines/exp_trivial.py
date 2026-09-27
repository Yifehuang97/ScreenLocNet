"""Input-free baselines.

Mean-Train predicts the mean training screen for every clip. Oracle-Const is the
single rectangle minimising *test* error, so it upper-bounds any predictor that
ignores its input. Both are far worse than the network, which is the point.
"""
import os
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("SCRLOC_RESULTS", "baselines/results")
os.makedirs(RESULTS, exist_ok=True)
import numpy as np, common
from train_baselines import build

out = {}
for scene in ["2", "3"]:
    d = build(scene)
    tr, te = d["train"]["corners"], d["test"]["corners"]
    row = {}
    for name, const in [("Mean-Train", tr.mean(0)),
                        ("Median-Train", np.median(tr, axis=0)),
                        ("Oracle-Const", te.mean(0))]:
        pred = np.repeat(const[None], len(te), axis=0)
        r = common.evaluate(pred, te)
        r.pop("_corner_per_clip"); r.pop("_center_per_clip")
        row[name] = r
        print(f"scene {scene} {name:13s} Center {r['Center-Dist']:6.2f}  Corner {r['Corner-Dist']:6.2f}  "
              f"VP-IoU {r['VP-IoU']:.4f}")
    # spread of the GT screens: how much variation is there to explain at all?
    row["_gt_spread_cm"] = float(np.sqrt(((te - te.mean(0)) ** 2).sum(-1)).mean() * 100)
    print(f"scene {scene} GT corner spread about the test mean: {row['_gt_spread_cm']:.2f} cm\n")
    out[scene] = row

json.dump(out, open(os.path.join(RESULTS, 'results_trivial.json'), 'w'), indent=1)
print("saved ->", os.path.join(RESULTS, "results_trivial.json"))
