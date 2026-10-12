# UNet-Coastlines

Detects Earth's coastline in synthetic spacecraft-camera images using a HED-UNet
(a UNet that predicts a per-pixel class map and a coastline edge map at the same time).
Images are rendered with POV-Ray from simulated spacecraft positions, so ground truth is exact.

This README describes the pipeline **as it works now**. Run everything from the repo root.

## Quick start

All settings live in `src/config.py`. Run from the repo root.

```
python src/run_pipeline.py                      # train + evaluate with the settings in config.py
python src/run_pipeline.py --stages generate,labels,val-labels,train,evaluate    # everything from scratch
python src/run_pipeline.py --set IMG_SIZE=256 --set EPOCHS=50                    # one run, a few settings changed
python src/run_pipeline.py --sweep IMG_SIZE=256,512 --sweep STACK_HEIGHT=4,5     # every combination, compared in runs/results.csv
python src/run_pipeline.py --sweep TRAIN_FRACTION=0.25,0.5,1.0                   # learning curve: same validation set, less training data
python src/evaluate_hed_unet.py --run <run name>                                 # re-score a finished run (updates its metrics.json and plots, not results.csv)
python src/plot_runs.py                                                          # loss curves of every run on one plot
python src/predict_hed_unet.py 23 --run <run name>                               # look at one validation image
```

Each run writes to its own folder `runs/<run name>/` (model, checkpoints, `config.json`, `train_log.csv`,
`metrics.json`, `validation_results/` overlay plots). The run name is built from the settings you
overrode (e.g. `img256_sh4_lr0.0003`: image size 256, stack height 4, learning rate 3e-4), so runs never overwrite each other. Flag order doesn't matter, and the full settings are in each run's `config.json` and in the `overrides` column of `results.csv`. `runs/results.csv`
has one row per run (settings + validation metrics) for comparing them.

## Pipeline

The dataset stages (`generate`, `labels`, `val-labels`) run **once** at the start, even in a `--sweep`, so every run trains on identical images. Resizing to `IMG_SIZE` happens when images are loaded, so one set of full-size images and labels serves every resolution.

```
1. render scenes        src/main_image_gen_SC_coast_local.py  (uses src/functions.py)      stage: generate
        |  earth_img_<N>.png + 4 flat renders per view, written to RENDER_DIR
2. build labels         src/make_label_masks.py [validation] [--archive]                   stages: labels, val-labels
        |  dataset/images + dataset/labels  (or dataset/validation_*), used renders archived
3. train                src/train_hed_unet.py          -> runs/<run>/model.pth             stage: train
4. score validation set src/evaluate_hed_unet.py       -> runs/<run>/metrics.json + plots  stage: evaluate
5. look at one image    src/predict_hed_unet.py [number] [--run NAME]
```

`--run <name>` for evaluate/predict loads that run's saved `config.json`, so resolution, depth and classes match its weights.

Validation images come from `dataset/validation_*` (`VAL_SOURCE = "folder"`), or `VAL_FRACTION` of the
training pool is held out (`VAL_SOURCE = "split"`, reproducible via `SPLIT_SEED`). To compare runs fairly keep the validation set fixed and vary `TRAIN_FRACTION` (how much of the training set is used), not `VAL_FRACTION`.

### 1. Rendering (`main_image_gen_SC_coast_local.py`, `functions.py`)
- Picks spacecraft positions (theta / phi / distance) and a date, then calls `gen_moon_earth`
  to render the normal RGB image `earth_img_<N>.png` (Earth map + cloud map + sun + atmosphere in POV-Ray).
- Then calls `gen_earth_label_renders`, which renders four flat "ingredient" images from the
  **same camera, Earth rotation, cloud map and sun**:

  | file | what it is |
  |---|---|
  | `earth_img_LANDFLAT<N>.png` | unlit land/ocean map (ocean black, land colored, night side included) |
  | `earth_img_CLOUDFLAT<N>.png` | unlit cloud coverage, brightness = coverage 0-1 |
  | `earth_img_DISK<N>.png` | Earth as a flat white disk (white = Earth, black = space) |
  | `earth_img_LIT<N>.png` | plain white Earth lit by the sun (black = night side) |

- Needs SPICE kernels in `Kernels/` and the Earth/cloud maps in `Maps/` (both git-ignored).
- Camera used for error units (`case_type` 3000): focal length 35 mm, pixel size 4.96e-3 mm (`FOCAL_LEN_MM`, `PIXEL_SIZE_MM` in `config.py`).

### 2. Labels (`make_label_masks.py`)
Combines the four renders into one single-channel PNG per image, where each pixel value is a class:

| value | class | rule (thresholds are 0-1 brightness) |
|---|---|---|
| 0 | space | outside the DISK render |
| 1 | water | default |
| 2 | land | LANDFLAT > 0.08 |
| 3 | cloud | CLOUDFLAT > 0.4 |
| 4 | dark (night side) | LIT < 0.1 |

Priority when classes overlap: space > dark > cloud > land/water.
`python src/make_label_masks.py` writes the training set, `... validation` writes the validation set.
It processes **every** render set in `RENDER_DIR`; `--archive` (used by `run_pipeline.py`) moves the used
renders to `dataset/render_archive/` so the next batch starts clean. Views with a missing render are skipped. Color previews go to `dataset/*label_previews/`.

The coastline is **not stored**. It is derived at train time: a land pixel touching water, or
water touching land (3x3 neighborhood). Cloud, dark and space boundaries are never coastline.

### 3. Training (`train_hed_unet.py`, `hed_unet.py`, `coastline_dataset.py`)
- **Settings** all come from `src/config.py` (`IMG_SIZE`, `STACK_HEIGHT`, `BASE_CHANNELS`, `MERGE_DARK`,
  `EPOCHS`, `PATIENCE`, `BATCH_SIZE`, `LEARNING_RATE`, `MAX_COASTLINE_WEIGHT`, `SEED`, split settings, paths, camera).
  Override any of them with `--set NAME=value`. `IMG_SIZE` is checked against `2**STACK_HEIGHT` up front.
- **Input handling:** `letterbox()` shrinks the long side to `IMG_SIZE` without stretching and pads
  with black, so geometry is preserved. Labels use nearest-neighbor so class numbers are never blended.
  The scale and padding are kept to map predictions back to original pixels.
- **`MERGE_DARK = True`** treats night side (4) as space (0) because both are pure black in the image;
  the model could only separate them by disk shape, which failed at 512 res. Output classes become
  `space/dark, water, land, cloud` (4). Label files still hold all 5 classes.
- **Model:** UNet encoder/decoder (strided-conv downsampling, transpose-conv upsampling, batch norm,
  skip connections), `stack_height` levels, base 16 channels. Each decoder level has its own head
  (class scores + 1 coastline channel), and the levels are merged with a learned per-pixel
  attention (softmax across levels). `IMG_SIZE` must be divisible by `2**STACK_HEIGHT`.
- **Loss:** cross-entropy on classes + class-balanced BCE on the coastline channel, summed over the
  merged output and every level (deep supervision). Coastline is only ~1-2% of pixels, so the positive
  weight is capped at 10x (full balancing, ~65x, made the model mark nearly the whole Earth as coast); set by `MAX_COASTLINE_WEIGHT`.
- **Early stopping** (`PATIENCE`, default 10; 0 = off): training stops once the validation loss has not improved for `PATIENCE` epochs,
  and the saved `model.pth` is the model from the BEST epoch, not the last. `EPOCHS` is then only a ceiling, so every setting gets the
  training length it needs. `epochs_run`, `stopped_early` and `best_val_epoch` are recorded in `results.csv`.
  Note the validation set is used both to stop and to rank runs, so the numbers are slightly optimistic; report the final
  winner on images that were never used for either (a separate test batch).
- Saves a checkpoint every `CHECKPOINT_EVERY` epochs and the best-epoch weights (last epoch if early stopping is off) to `runs/<run>/model.pth`.
- **A trained `.pth` only works at the `IMG_SIZE`/`MERGE_DARK`/`STACK_HEIGHT`/`BASE_CHANNELS` it was trained with.** Each run folder stores its `config.json` so you can check.

### 4-5. Evaluation and prediction
- `evaluate_hed_unet.py` runs the whole validation set and writes an overlay per image, `summary.csv` and `metrics.json`.
- `predict_hed_unet.py <number>` shows one validation image: class maps, coastline and error numbers.
- **Coastline post-processing** (`coastline_lines.py`): threshold the edge probability at 0.5 ->
  skeletonize to a 1-pixel line -> move each point to the probability-weighted center of its
  neighborhood (sub-pixel) -> trace into ordered point lists -> map back to original image pixels.
- **Metrics** (saved to `runs/<run>/metrics.json` and one row of `runs/results.csv`):
  - Coastline error: distance from each predicted line point to the nearest true land/water boundary point,
    in ORIGINAL pixels, so it is comparable across `IMG_SIZE`: `coast_error_median_px`, `_mean_px`, `_p90_px`,
    `_median_mrad`, `_p90_mrad` (1 original pixel = 0.142 mrad).
  - `coast_found_fixed_pct`: % of true coastline points with a predicted point within `FOUND_TOLERANCE_PX`
    original pixels (comparable across resolutions). `coast_found_pct` uses 1 *model* pixel, so only compare it between runs with the same `IMG_SIZE`.
  - Class scores at original resolution (prediction scaled back up, compared with the full-size label), comparable across
    resolutions: `full_pixel_accuracy`, `full_mean_iou`, `full_iou_<class>`.
    `pixel_accuracy`, `mean_iou`, `iou_<class>` are the same scores at the model's own resolution.
  - Cost: `n_params`, `infer_ms_median` (forward pass per image), `train_minutes`.
  - Training: `final_loss`, `final_val_loss`, `best_val_loss`, `best_val_epoch` (loss on the validation images, logged only, never trained on).
- **Loss curves:** `runs/<run>/train_log.csv` (epoch, avg_loss, val_loss, seconds) and `runs/<run>/loss_curve.png` are written for every run.
  `python src/plot_runs.py` overlays all runs; `python src/plot_runs.py --metric coast_error_median_px --vs TRAIN_FRACTION`
  plots a metric against a swept setting (the learning curve). `--where KEY=VALUE` narrows to runs with that setting.

## Repo layout

| path | role |
|---|---|
| `src/config.py` | **every setting**; `--set NAME=value` overrides it from the command line |
| `src/run_pipeline.py` | runs the stages end to end, single run or sweep |
| `src/functions.py` | rendering, orbital/geodesy math, edge detection, older coastline functions (very large) |
| `src/main_image_gen_SC_coast_local.py` | render driver for the ML data |
| `src/make_label_masks.py` | renders -> label maps |
| `src/coastline_dataset.py` | `letterbox`, train/validation split, `CoastlineLabelDataset` |
| `src/plot_runs.py` | loss curves and metric-vs-setting plots |
| `src/hed_unet.py`, `train_hed_unet.py`, `evaluate_hed_unet.py`, `predict_hed_unet.py`, `coastline_lines.py` | HED-UNet model + workflow |
| `src/main.py`, `main_image_gen_SC_centroiding.py`, `preprocess_earth_local.py` | older generation / centroiding scripts, not used by the ML pipeline |
| `training_earth.yaml`, `digitalized_points.csv` | not used by the current pipeline |

(The plain-UNet scripts were removed; they are in git history before the cleanup commit.)

## Requirements
Python with `torch`, `torchvision`, `numpy`, `pillow`, `scipy`, `scikit-image`, `matplotlib`
(ML side); plus `spiceypy`, `opencv-python` and POV-Ray for rendering.

## References
Cited in the code:
- Heidler, K., et al. (2021). *HED-UNet: Combined Segmentation and Edge Detection for Monitoring
  the Antarctic Coastline.* The model in `src/hed_unet.py` is adapted from
  https://github.com/khdlr/HED-UNet (MIT License, Copyright (c) 2021 Konrad Heidler).
- Cloud maps: https://www.shadedrelief.com/natural3/pages/clouds.html
- Ephemerides: JPL HORIZONS, https://ssd.jpl.nasa.gov/horizons/app.html
- Moon radius: NASA SVS, https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
- POV-Ray rotation matrix usage: https://www.povray.org/documentation/view/3.6.0/49/
- Atmosphere model: simple exponential model (link in `functions.py` near the `ATMO_STRENGTH` lines);
  starfield/scene include from https://github.com/pyramid3d/space3d
- Image noise: https://scikit-image.org/docs/stable/api/skimage.util.html#random-noise

Used but not cited in the code (add formal citations if you publish):
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical
  Image Segmentation* (the HED-UNet backbone).
- NAIF SPICE toolkit / `spiceypy` and the DE441 ephemeris kernels loaded from `Kernels/`.
