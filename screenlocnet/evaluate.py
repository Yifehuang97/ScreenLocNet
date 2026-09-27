"""Evaluate a checkpoint.

    python -m screenlocnet.evaluate --data /path/to/ScrLoc30Exp \
        --test-scene 2 --checkpoint checkpoint.pth
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import ScreenLocClips, build_splits
from .metrics import summarize
from .model import ScreenLocNet
from .train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-scene", type=int, default=2, choices=[2, 3])
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, te_ids = build_splits(a.data, a.test_scene)
    loader = DataLoader(ScreenLocClips(a.data, te_ids), batch_size=4, num_workers=4)

    model = ScreenLocNet(a.dim, a.layers, a.heads).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    pred, gt, rec = predict(model, loader, device)
    print(json.dumps(summarize(pred, gt, rec), indent=2))


if __name__ == "__main__":
    main()
