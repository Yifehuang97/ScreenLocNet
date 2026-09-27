# ScreenLocNet

**Display Screen Localization from Gaze Directions** — ACCV 2026

Given a webcam video of someone working at a screen, ScreenLocNet predicts the 3D
positions of the screen's four corners in the camera frame. The screen is never
visible to the camera and nothing is calibrated beforehand.

The cue is that a viewer's gaze rays sweep the screen while they use it, so the angular
support of the ray bundle constrains where the screen must be. Absolute scale comes
from DUSt3R's point map, which is anchored by the human face.

Accuracy is about 15 cm of screen-centre error on a scene never seen in training,
which is already enough to tell whether someone is looking at their screen.

## Install

```bash
git clone https://github.com/Yifehuang97/ScreenLocNet && cd ScreenLocNet
pip install -r requirements.txt
```

## Data

```bash
python scripts/download_data.py --out /path/to/ScrLoc30Exp
```

ScreenLoc is 151 recording sessions from 34 subjects across 3 scenes, cut into 2,802
clips of 1,200 face-valid frames. Each clip ships as one `.npz` with the per-frame
gaze rays, the ground-truth screen corners, and the recording and session ids.

Ground truth came from an auxiliary camera with AprilTag markers at the screen corners.
Its calibration reprojection error has a median of 2.4 px, far below the errors the
method itself makes.

## Train and evaluate

```bash
python -m screenlocnet.train    --data /path/to/ScrLoc30Exp --test-scene 2
python -m screenlocnet.evaluate --data /path/to/ScrLoc30Exp --test-scene 2 \
                                --checkpoint checkpoint.pth
```

`--test-scene 2` and `--test-scene 3` are the paper's two cross-scene settings; they
deliberately use different training sets. See [docs/reproducing.md](docs/reproducing.md)
for which is which, the baselines, and how to report error without flattering yourself.

## Layout

```
screenlocnet/
  model.py       the network. Two branches, one 8-DOF head
  geometry.py    the rectangle parameterisation; 8 DOF, not 12
  data.py        clips and the two cross-scene splits
  metrics.py     Corner-Dist, Center-Dist, VP-IoU, recording-level bootstrap
  train.py       training entry point
  evaluate.py    evaluation entry point
baselines/       input-free, gaze-only and geometric-fitting comparisons
scripts/         data download
docs/            reproduction notes and the full baseline table
```

## Two things that are easy to get wrong

**The architecture is not really temporal.** No positional encoding is added to either
transformer and the pooling is a mean, so frame order enters only through local Conv1d
windows. Shuffling the frames costs 11.22 to 11.99 cm. A matched-capacity
order-sensitive control behaves the same way. Read the gaze branch as a multi-scale
*set* encoder over gaze rays.

**Clip-level error bars are too narrow.** Clips from one recording overlap by up to 600
frames. Use `metrics.recording_bootstrap`, which resamples whole recordings; it roughly
quadruples the interval and is the honest number.

## Citation

```bibtex
@inproceedings{huang2026screenlocnet,
  title     = {ScreenLocNet: Display Screen Localization from Gaze Directions},
  author    = {Huang, Yifeng and Nguyen, Huy Anh and Chandran, Prasanth and
               D'Mello, Sidney K. and Rebello, N. Sanjay and Loschky, Lester and
               Hoai, Minh},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```
