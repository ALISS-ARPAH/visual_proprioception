# Results of the shipped cut-depth model

All files here come from [models/cut_depth_combined](../models/cut_depth_combined/). The `.mp4` videos are git-ignored because of their size (up to 80 MB each). They exist only on the machine they were rendered on, and each one can be regenerated from its recording (commands below).

## unseen/: recordings the model never saw

The training sessions were recorded 7–15 Oct 2026; all four recordings below came later. Each was scored with `score_contact_bag`, which processes frames exactly as the live node does.

| Recording | Recorded | Length | Firings | Flagged (P ≥ 0.8) in the 1 s before firing | Laser-off frames flagged |
|---|---|---|---|---|---|
| `lesion_2_cut_1` | 21 Oct 2026 | 813 s | 50 | 37 (74%) | 10% |
| `cut_1` | 29 Oct 2026 | 513 s | 8 | 6 (75%) | 20% |
| `marking` | 29 Oct 2026 | 323 s | 18 | 13 (72%) | 20% |
| `proprioception_w_slam` | 30 Oct 2026 | 167 s | 0 | – | 1% |

- **lesion_2_cut_1:** split files 17 and 18 were empty and 15 and 16 were not finalised, so the end of the recording is missing.
- **proprioception_w_slam:** no firings. The fiber is out of view for the first ~60 s, and the camera stream has gaps of 9–19 s.

Each recording has three files:
- **`<name>_prediction.mp4`:** the camera frame with the detected laser spot outlined in red, the crop box and the projected tip, plus the last 10 s of P against the 0.6 / 0.8 thresholds.
- **`<name>_prediction.csv`:** one row per frame:
  - `p_raw`: the model output
  - `p`: the 0.3 s mean that the decision uses
  - `decision`
  - `laser`
  - `spot_detected`
  - `crop_source`: spot, tip or last spot
  - `tip_u` / `tip_v`: the projected tip
- **`<name>_prediction.log`:** the run summary.

Regenerate a video and CSV:

```bash
ros2 run visual_proprioception score_contact_bag --bag <recording folder> \
  --model models/cut_depth_combined/model.json --out results/unseen/<name>_prediction.mp4
```

The source recordings were:
- `/media/james/Backup Plus/lesion_2/cut_1`
- `/media/james/Backup Plus/proprioception_w_slam`
- `~/aliss_runs/test_data/cut_1`
- `~/aliss_runs/test_data/marking`

## heldout/: training sessions, cross-validated

`cut_depth_combined_heldout.mp4` shows 12 approaches from all five training sessions, back to back. Each approach is scored by the cross-validation model that didn't train on its bladder. The video was made with `render_contact_video --run-dir ~/aliss_runs/contact_full/runD_seed0 --model combined`.

## colleague_lesion_2/: head-to-head with BladderTouchDynamicCrop

This compares our model and the colleague's on the same frames of `lesion_2_cut_1`. The labels are depth labels built from the tip pose with the training rule: within 1 mm of the cutting depth = yes, 2 mm or more short = no.

- **`both_models.csv`:** per-frame scores from both models plus the tip pose (`scripts/colleague/score_bag.py`).
- **`evaluation.txt`:** the full comparison (`scripts/colleague/evaluate_bag.py both_models.csv`).

| Measure | Colleague | Ours |
|---|---|---|
| AUC at 1 mm (yes vs no frames) | 0.73 | 0.90 |
| AUC laser-on vs 2 mm or more short | 0.83 | 0.90 |
| Firings flagged in the 1 s before, own threshold | 45/50 (90%) at 0.5 | 37/50 (74%) at 0.8 |
| Far frames flagged, own threshold | 35% | 5% |
| Firings flagged at a matched 10% false-alarm rate | 42% | 88% |

The AUC difference at 1 mm is +0.17 (95% interval over firings +0.07 to +0.29).

Two caveats:
- The labels are the target our model was trained on, which favours it.
- This is one recording. The colleague's model needs the projected tip in the image, so both models are compared on the 81% of frames where it has a score.
