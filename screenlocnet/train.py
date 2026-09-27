"""Train ScreenLocNet.

    python -m screenlocnet.train --data /path/to/ScrLoc30Exp --test-scene 2

Reproduces the setting behind Tables 1 and 2: Adam at 1e-5, StepLR halving every 5
epochs, batch 4, 10 epochs, MSE on the four corners, and the epoch chosen on the
validation split (never on test).
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ScreenLocClips, build_splits
from .metrics import corner_dist, summarize
from .model import ScreenLocNet


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gts, recs = [], [], []
    for b in loader:
        out = model(b["gaze_origin"].to(device), b["gaze_dir"].to(device),
                    b["pointmap"].to(device), b["dino_face"].to(device))
        preds.append(out.cpu().numpy())
        gts.append(b["corners"].numpy())
        recs.append(b["recording_id"].numpy())
    return (np.concatenate(preds).reshape(-1, 4, 3),
            np.concatenate(gts).reshape(-1, 4, 3),
            np.concatenate(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root of the ScrLoc30Exp download")
    ap.add_argument("--test-scene", type=int, default=2, choices=[2, 3])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3421)
    ap.add_argument("--out", default="checkpoint.pth")
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_ids, va_ids, te_ids = build_splits(a.data, a.test_scene)
    print(f"train {len(tr_ids)}  val {len(va_ids)}  test {len(te_ids)} clips")
    mk = lambda ids, sh: DataLoader(ScreenLocClips(a.data, ids), batch_size=a.batch_size,
                                    shuffle=sh, num_workers=4)
    train_loader, val_loader, test_loader = mk(tr_ids, True), mk(va_ids, False), mk(te_ids, False)

    model = ScreenLocNet(a.dim, a.layers, a.heads).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()

    best_val, best = float("inf"), None
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for b in train_loader:
            loss = criterion(
                model(b["gaze_origin"].to(device), b["gaze_dir"].to(device),
                      b["pointmap"].to(device), b["dino_face"].to(device)),
                b["corners"].to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(b["corners"])
        sched.step()

        vp, vg, _ = predict(model, val_loader, device)
        val = corner_dist(vp, vg).mean()
        print(f"epoch {epoch:02d}  train MSE {total/len(tr_ids):.5f}  val Corner-Dist {val:.2f} cm")
        if val < best_val:
            best_val, best = val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best, a.out)

    model.load_state_dict(best)
    tp, tg, tr_rec = predict(model, test_loader, device)
    print(f"\nScene {a.test_scene} test results")
    print(json.dumps(summarize(tp, tg, tr_rec), indent=2))


if __name__ == "__main__":
    main()
