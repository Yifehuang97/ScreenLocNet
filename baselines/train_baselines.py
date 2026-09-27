"""Gaze-only baselines at matched capacity.

Four ways of aggregating the same 1,200 gaze rays, all with the same 8-DOF head,
loss, splits, optimiser and schedule:

    deepsets   permutation-invariant, mean and max pooling
    density    permutation-invariant, 2D histogram of gaze directions into a CNN
    settx      permutation-invariant, Set Transformer (ISAB + PMA)
    temporal   order-sensitive, multi-scale Conv1d windows plus positional encoding

Pass --shuffle to destroy frame order, which isolates what order is worth.
"""
import os
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
RESULTS = os.environ.get("SCRLOC_RESULTS", "baselines/results")
os.makedirs(RESULTS, exist_ok=True)
import common
from gaze_only_models import GazeBaseline

DIMS = {"temporal": 256, "settx": 320, "deepsets": 512, "density": 640}


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Clips(torch.utils.data.Dataset):
    def __init__(self, d, n_frames=1200, shuffle_frames=False):
        self.d, self.n, self.sh = d, n_frames, shuffle_frames

    def __len__(self):
        return len(self.d["gaze"])

    def __getitem__(self, i):
        g = self.d["gaze"][i]
        if self.n < g.shape[0]:
            g = g[:self.n]                       # leading sub-window
        if self.sh:
            g = g[np.random.permutation(g.shape[0])]
        f = np.zeros_like(g)                     # bg_ssl stores a constant origin
        return torch.from_numpy(f), torch.from_numpy(g), \
               torch.from_numpy(self.d["corners"][i].reshape(-1))


@torch.no_grad()
def infer(model, ds, dev, bs=16):
    model.eval()
    out = []
    for f, g, _ in torch.utils.data.DataLoader(ds, batch_size=bs):
        out.append(model(f.to(dev), g.to(dev)).cpu().numpy())
    return np.concatenate(out).reshape(-1, 4, 3)


def run(kind, scene, data, dev, seed=3421, epochs=25, lr=1e-4, bs=16,
        n_frames=1200, shuffle_frames=False, log=print):
    set_seed(seed)
    tr = Clips(data["train"], n_frames, shuffle_frames)
    va = Clips(data["val"], n_frames, shuffle_frames)
    te = Clips(data["test"], n_frames, shuffle_frames)
    model = GazeBaseline(kind, dim=DIMS[kind]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(5, epochs // 3), gamma=0.5)
    loader = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, drop_last=False)
    crit = nn.MSELoss()

    best_val, best_pred = 1e9, None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for f, g, y in loader:
            f, g, y = f.to(dev), g.to(dev), y.to(dev)
            loss = crit(model(f, g), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(f)
        sch.step()
        vp = infer(model, va, dev)
        v = common.corner_dist(vp, data["val"]["corners"]).mean()
        if v < best_val:
            best_val = v
            best_pred = infer(model, te, dev)
        log(f"    [{kind}|{scene}] ep{ep:02d} loss {tot/len(tr):.5f} val {v:.2f} best {best_val:.2f}")

    res = common.evaluate(best_pred, data["test"]["corners"])
    res["val_corner"] = float(best_val)
    res["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    return res, best_pred


def build(scene, cache={}):
    """scene '2' -> test split scene_1 (425);  scene '3' -> test split scene_2 (324)."""
    if scene in cache:
        return cache[scene]
    tr_ids = common.load_split("scene_1", "train")
    va_ids = common.load_split("scene_1", "val")
    if scene == "2":
        te_ids = common.load_split("scene_1", "test")
        tr_ids = tr_ids + common.load_split("scene_2", "test")   # + Scene-3 clips
    else:
        te_ids = common.load_split("scene_2", "test")
    print(f"  scene {scene}: train {len(tr_ids)} val {len(va_ids)} test {len(te_ids)}", flush=True)
    d = {k: common.load_clips(v) for k, v in
         [("train", tr_ids), ("val", va_ids), ("test", te_ids)]}
    cache[scene] = d
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="temporal,settx,deepsets,density")
    ap.add_argument("--scenes", default="2,3")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", default="3421")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle frame order (isolates the contribution of temporal order)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(RESULTS, "results_baselines.json"))
    a = ap.parse_args()

    dev = torch.device("cuda:0")   # pinned to GPU 3 via CUDA_VISIBLE_DEVICES
    all_res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for scene in a.scenes.split(","):
        data = build(scene)
        for kind in a.kinds.split(","):
            for seed in [int(s) for s in a.seeds.split(",")]:
                key = f"scene{scene}|{kind}{a.tag}|{seed}"
                if key in all_res:
                    print("skip", key); continue
                t0 = time.time()
                r, pred = run(kind, scene, data, dev, seed=seed, epochs=a.epochs,
                              shuffle_frames=a.shuffle)
                r["seconds"] = time.time() - t0
                np.save(os.path.join(RESULTS, f"pred_{scene}_{kind}{a.tag}_{seed}.npy"), pred)
                r.pop("_corner_per_clip"); r.pop("_center_per_clip")
                all_res[key] = r
                print(f"== {key}: Center {r['Center-Dist']:.2f} Corner {r['Corner-Dist']:.2f} "
                      f"VP {r['VP-IoU']:.4f} ({r['seconds']:.0f}s)", flush=True)
                json.dump(all_res, open(a.out, "w"), indent=1)
