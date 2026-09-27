"""Fetch the preprocessed ScreenLoc clips.

    python scripts/download_data.py --out /path/to/ScrLoc30Exp

Downloads bg_ssl/npz (one .npz per clip: gaze rays, screen corners, recording and
session ids) and bg_ssl/final_splits. Roughly 0.6 GB.
"""
import argparse

from huggingface_hub import snapshot_download

REPO = "yifehuang97/ScrLoc30Exp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    path = snapshot_download(REPO, repo_type="dataset", local_dir=a.out,
                             max_workers=a.workers)
    print("downloaded to", path)


if __name__ == "__main__":
    main()
