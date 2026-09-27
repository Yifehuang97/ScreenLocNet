# Reproducing the paper

## What the two tables are

They do not share a training set, which the submission did not state clearly.

| Paper table | Test scene | Trains on | Clips | Reported Corner-Dist |
|---|---|---|---|---|
| Table 1 | Scene 2 | Scene 1 + Scene 3 | 2,055 | 17.50 cm |
| Table 2 | Scene 3 | Scene 1 only | 1,731 | 14.91 cm |

Table 2 is therefore the stricter result: a single training scene generalising to a
wholly unseen one. All splits are disjoint at the recording level, 88/18/22/23
recordings for train/val/Scene 2/Scene 3, so overlapping clips never straddle a split.

```bash
python -m screenlocnet.train --data $DATA --test-scene 2
python -m screenlocnet.train --data $DATA --test-scene 3
```

## Reporting error honestly

Clips from one recording overlap by up to 600 frames, so a clip-level confidence
interval is roughly four times too narrow. `metrics.recording_bootstrap` resamples
whole recordings instead, and it is what `summarize` reports:

| | Scene 2 | Scene 3 |
|---|---|---|
| Center-Dist mean / median / p90 | 15.73 / 14.35 / 25.99 | 13.45 / 13.36 / 19.57 |
| clip-level 95% CI | [14.9, 16.6] | [12.9, 14.0] |
| **recording-level 95% CI** | **[12.5, 20.0]** | **[11.8, 15.2]** |

Large errors are not spread evenly. All 31 Scene-2 clips above 30 cm come from 4 of 22
recordings, and all 4 such Scene-3 clips from 2 of 23. They are sessions where the
viewer barely moves and looks at a narrow region, so the gaze bundle under-constrains
the screen.

## Baselines

`baselines/` holds the comparisons added during review.

```bash
python baselines/exp_trivial.py        # input-free constants
python baselines/train_baselines.py    # gaze-only models at matched capacity
python baselines/exp_analysis.py       # percentiles and the cluster bootstrap
```

Corner-Dist in cm, mean over 3 seeds:

| | Scene 2 | Scene 3 |
|---|---|---|
| Mean-Train (constant) | 42.38 | 36.15 |
| Oracle-Const (best possible constant) | 36.44 | 33.38 |
| Robust geometric fit | 64.41 | 55.62 |
| DeepSets | 20.80 | 21.01 |
| Gaze-density CNN | 19.77 | 21.18 |
| Set Transformer | 20.30 | 22.13 |
| Multi-scale temporal (+positional encoding) | 19.50 | 23.19 |
| ...the same, with frames shuffled | 19.56 | 22.12 |
| **ScreenLocNet** | **17.50** | **14.91** |

Two things are worth reading off this table. Shuffling the frames of the
order-sensitive control changes nothing, and every gaze-only model lands in a
20.5-21.3 cm band regardless of architecture, so temporal order is not what the gaze
branch contributes. But no gaze-only model closes the gap to the full network either,
which is the scene-static branch earning its place.

## Timing

DUSt3R runs on 64 sampled frames per clip, not on all 1,200. End to end on one A5000:
16.1 s for DUSt3R, 5.7 s for L2CS at 4.79 ms per frame, 0.04 s for the forward pass,
so about 22 s per clip. Accuracy is flat in the window length,
20.7/20.2/20.1/20.5/20.1 cm at T = 1200/900/600/300/150, so the window can be
shortened eightfold.
