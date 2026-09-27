"""ScreenLoc dataset.

151 recording sessions from 34 subjects across 3 scenes, segmented into 2,802 clips of
1,200 face-valid frames. Adjacent clips may overlap by up to 600 frames, so anything
computed per clip is correlated; evaluate at the recording level, not the clip level.

The two split files define the two cross-scene settings of the paper:

    scene_1_split.json   test split = the paper's Scene 2   (425 clips, 22 recordings)
    scene_2_split.json   test split = the paper's Scene 3   (324 clips, 23 recordings)

train and val are identical in both files (1,731 / 322 clips, Scene 1).
The two tables of the paper do NOT share a training set:

    Table 1 (test Scene 2)  trains on Scene 1 + Scene 3   2,055 clips
    Table 2 (test Scene 3)  trains on Scene 1 only        1,731 clips

`build_splits` reproduces exactly that.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ScreenLocClips(Dataset):
    def __init__(self, root, clip_ids, load_features=True):
        self.root = root
        self.ids = list(clip_ids)
        self.load_features = load_features

    def __len__(self):
        return len(self.ids)

    def _npz(self, cid):
        return np.load(os.path.join(self.root, "bg_ssl", "npz", f"{cid:05d}.npz"))

    def __getitem__(self, i):
        cid = self.ids[i]
        d = self._npz(cid)
        item = {
            "gaze_origin": torch.from_numpy(d["face"].astype(np.float32)),
            "gaze_dir": torch.from_numpy(d["gaze_dir"].astype(np.float32)),
            "corners": torch.from_numpy(
                np.stack([d["tl"], d["tr"], d["bl"], d["br"]]).astype(np.float32).reshape(-1)),
            "recording_id": int(str(d["id"])),
            "session_id": int(str(d["session_id"])),
            "clip_id": cid,
        }
        if self.load_features:
            # 64 sampled frames were extracted; the model consumes every second one
            pm = np.load(os.path.join(self.root, "pts3d", f"face_{cid:05d}.npy"))
            df = np.load(os.path.join(self.root, "face_dino_v2_32", f"{cid:05d}.npy"))
            item["pointmap"] = torch.from_numpy(pm[::2].astype(np.float32))
            item["dino_face"] = torch.from_numpy(df[::2, 0, :].astype(np.float32))
        return item


def _split(root, name, part):
    with open(os.path.join(root, "bg_ssl", "final_splits", f"{name}_split.json")) as f:
        return json.load(f)[part]


def build_splits(root, test_scene):
    """test_scene is 2 or 3, matching the paper's table numbering."""
    if test_scene not in (2, 3):
        raise ValueError("test_scene must be 2 or 3")
    train = _split(root, "scene_1", "train")
    val = _split(root, "scene_1", "val")
    if test_scene == 2:
        test = _split(root, "scene_1", "test")
        train = train + _split(root, "scene_2", "test")   # Scene 3 clips join training
    else:
        test = _split(root, "scene_2", "test")
    return train, val, test
