# HED UNet-Coastlines: Welcome

Hi! This project teaches a neural network to find **coastlines in pictures of Earth taken from a spacecraft**. The pictures are not real photos. We render them on a computer, so we know exactly where every coastline is, which gives us perfect "answers" to train and test against.

You don't need to understand every file to contribute. This page tells you what to expect, how to run things, and where people get stuck. For the full technical details (labels, model, loss, metrics), see [README.md](README.md) in this folder.

## What you should know first
- **Python basics.** If you can read a script and run it from a terminal, you're fine.
- **You don't need deep-learning experience.** The sections below explain the ideas you'll meet.
- **Don't panic about `src/functions.py`.** It's about 10,000 lines of inherited rendering and orbit code. You'll almost never need to read it. Please don't try to refactor it.
- **Some things take hours.** Rendering images and training sweeps are slow. Plan to start a run, then do something else.

## The big picture
```
1. Render images       (slow, needs POV-Ray)    -> pictures + helper images
2. Make labels         (fast)                   -> what each pixel is: space / water / land / cloud
3. Train the model     (minutes to hours)       -> a saved model file
4. Evaluate            (fast)                   -> numbers + pictures showing how well it did
```
One script, `src/run_pipeline.py`, runs these steps for you. Every setting you might want to change (image size, how long to train, and so on) lives in one file, `src/config.py`.

## Setting up
1. Get the code: `git clone -b HED_UNET_TEST https://github.com/BrendanUribe/UNet-Coastlines.git`
2. Install Python packages: `torch`, `torchvision`, `numpy`, `pillow`, `scipy`, `scikit-image`, `matplotlib`. For rendering you also need `spiceypy`, `opencv-python` and POV-Ray.
3. **Rendering needs two folders that are not in git:** `Kernels/` (SPICE data) and `Maps/` (Earth and cloud maps). Ask [YOUR MENTOR / LAB CONTACT] where to get them.
4. **GPU or no GPU?** Training works on a plain CPU but is slow, roughly a few minutes per epoch at 512 pixels. A GPU is about 10 seconds per epoch. Most NVIDIA GPUs work. AMD GPUs often don't work with PyTorch on Windows. If you have no usable GPU, use Kaggle (see below).

**Windows tip:** if `python` says "Python was not found", Windows is using a Microsoft Store shortcut. Try `py` instead, or fix it in Settings, Apps, Advanced app settings, App execution aliases.

## Your first run
Run everything from the repo root. This renders a small dataset, labels it, trains, and evaluates:
```
python src/run_pipeline.py --stages generate,labels,train,evaluate --set VAL_SOURCE=split --run test1
```
`VAL_SOURCE=split` keeps 20% of your images hidden from training and uses them for scoring. Without it, the pipeline expects a separate validation folder and will fail.

Results appear in `runs/test1/`. Look at `loss_curve.png`, `metrics.json`, and the pictures in `validation_results/`.

Already have images? Skip rendering and just train:
```
python src/run_pipeline.py --set VAL_SOURCE=split --run test2
```

### Before you render again
The generator **never deletes old files**. It writes into whatever folder you run it from. Old `earth_img_*.png` files get mixed into your new dataset, because the labeler processes everything it finds. Clear them first:
```
rm -f earth_img_*.png image_gen_data_*.npz
```
Also delete an old `dataset/` folder if you want a clean start.

## Changing settings and running experiments
Don't edit the code to try a new setting. Pass it on the command line:
```
python src/run_pipeline.py --set IMG_SIZE=256 --set EPOCHS=50
```
To try several values, and every combination of them, use `--sweep`:
```
python src/run_pipeline.py --set VAL_SOURCE=split --sweep LEARNING_RATE=3e-4,1e-3 --sweep BASE_CHANNELS=16,32
```
That makes 4 runs. Each gets its own folder in `runs/`, named after what you changed, and each adds one row to `runs/results.csv` so you can compare them. A sweep builds the dataset **once**, so every run uses identical images.

The settings you'll care about most:

| setting | what it does |
|---|---|
| `IMG_SIZE` | resolution the model sees (256, 512, 1024...). Must divide by `2**STACK_HEIGHT` |
| `STACK_HEIGHT` | how deep the network is |
| `BASE_CHANNELS` | how wide the network is |
| `BATCH_SIZE` | images processed per training step |
| `LEARNING_RATE` | how big each training step is. Too high is unstable, too low is slow |
| `EPOCHS`, `PATIENCE` | training stops after `EPOCHS`, or earlier once the validation loss stops improving for `PATIENCE` epochs. The saved model is the best one |
| `TRAIN_FRACTION` | train on only part of the training set (for "does more data help?") |
| `VAL_FRACTION`, `SPLIT_SEED` | share of images held out, and which ones |

## Ideas you'll run into
- **Training vs validation.** The model learns from the training images only. The validation images are held out, so the scores show how it does on pictures it hasn't seen. Never judge a model on its training images.
- **Overfitting.** The model gets better on training images while getting worse on validation images. The loss curve shows it. Early stopping helps.
- **Learning rate.** Think of it as how big a step the model takes each time it adjusts itself.
- **Seed.** A starting number for the random choices. Different seeds give slightly different results, so a tiny difference between two runs may just be noise.
- **HED-UNet.** The network we use. It predicts a class for every pixel *and* a coastline edge at several scales (Heidler et al., 2021).

## Reading the results
In `runs/results.csv`, these matter most:
- `coast_error_median_px`, `coast_error_p90_px`, `coast_error_median_mrad`: how far the predicted coastline is from the true one, in original image pixels. Smaller is better. One pixel is 0.142 milliradians. This is **the headline number**, and it can be compared across image sizes.
- `coast_found_fixed_pct`: how much of the true coastline the model found (within 2 original pixels). Bigger is better.
- `images_missed`: images where a coastline existed and the model drew none. These count as not found.
- `full_mean_iou`, `full_iou_land`, `full_iou_water`: how well the classes are found (0 to 1, bigger is better).
- `train_minutes`, `n_params`, `infer_ms_median`: what it cost.

**Don't compare `coast_found_pct` between different image sizes.** It uses a different tolerance at each size. Use `coast_found_fixed_pct` instead.
All the scores are pooled over every validation image, not averaged per image. With only about 20 validation images, small differences between runs are noise. Treat results as trends, not facts.

To draw pictures of your results:
```
python src/plot_runs.py                                                       # every run's loss curve
python src/plot_runs.py --metric coast_error_median_px --vs TRAIN_FRACTION    # a metric against one setting
```

## Where things stand
Our first test (74 training images, 19 validation, 512 pixels, default settings) gave:
- median coastline error **2.8 px (0.40 mrad)**, 90% of points under 14.4 px,
- mean IoU **0.82** (water 0.73, land 0.71, cloud 0.86),
- but only **about 42%** of the true coastline was found within 4 original pixels, and **18%** within 2.

So the model draws accurate coastline where it draws it, but misses a lot. Improving that is the main open problem. Validation loss was also very noisy, so a higher `PATIENCE` (around 20) is safer. Sweep results will be added here.

Ideas nobody has tried yet (not promises, just leads):
- a lower threshold for calling a pixel "coastline" (it's 0.5 now),
- a higher `MAX_COASTLINE_WEIGHT`,
- training on random crops of full-resolution images,
- a separate test set that is never used to tune anything,
- a larger and more varied dataset (more distances, dates, and cloud cover).

## Running on Kaggle (free GPU)
Kaggle gives about 30 GPU hours a week. You need a phone-verified account.
1. Upload `dataset/images` and `dataset/labels` as a **private Kaggle Dataset**.
2. Create a notebook. Turn on **GPU T4** and **Internet**, and add your dataset as an input.
3. Cells to paste:
   ```python
   import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
   ```
   ```
   !find /kaggle/input -maxdepth 4 -type d -name images
   ```
   ```python
   DATA = "<the path from the find command, without /images>"
   ```
   ```
   !git clone -b HED_UNET_TEST https://github.com/BrendanUribe/UNet-Coastlines.git
   %cd UNet-Coastlines
   ```
   ```
   !python src/run_pipeline.py --set DATA_DIR={DATA} --set VAL_SOURCE=split --set CHECKPOINT_EVERY=0 --set PATIENCE=20 --no-plots --sweep LEARNING_RATE=3e-4,1e-3
   ```
   ```
   !zip -r runs_light.zip runs -x "*.pth"
   ```
4. To walk away: **Save Version**, then **Save & Run All (Commit)**. It runs on Kaggle's machines. Download `runs_light.zip` from the version's Output tab when it finishes. That "commit" is Kaggle's, not git, and it doesn't touch GitHub.
5. A notebook run has a time limit (roughly 9 to 12 hours), so split big sweeps into chunks, and turn off your interactive session when you're done so it doesn't use up your hours.

## Memory (a rough guide)
GPU memory use grows with image size, roughly 4x per doubling. Rough figures for one image: about 0.7 GB at 512 pixels, 2.7 GB at 1024, 11 GB at 2048 (full resolution). More channels and bigger batches multiply that. If you see "out of memory" in a sweep, that run is recorded as an error and the sweep keeps going.

## Common problems
| problem | fix |
|---|---|
| `Python was not found` (Windows) | use `py`, or turn off the Store alias (see Setting up) |
| training is very slow | you're on CPU. Use a smaller `IMG_SIZE` or Kaggle |
| evaluation fails or shows NaN | no validation images. Use `--set VAL_SOURCE=split` or run the `val-labels` stage |
| strange or mixed images | old renders in the folder. Clear them before generating |
| `NameError: earth_map` from the generator | the render call must keep all three return values: `scene_file, earth_map, cloud_map = gen_moon_earth(...)` |
| VS Code: "clean your repository working tree" | you have uncommitted changes. Stash or commit them first |
| out of memory | smaller `IMG_SIZE`, `BATCH_SIZE`, or `BASE_CHANNELS` |

## Ground rules (suggested)
- Do a tiny run first (`--set EPOCHS=2`) to check that everything works before a long one.
- Never commit data, models, or run folders. `dataset/`, `runs/`, `*.pth` and the render PNGs are ignored on purpose.
- Change one thing at a time, or use `--sweep`. Write down what you tried.
- Keep the validation set fixed when you compare runs.
- Ask before changing `src/functions.py`.

## Where the code is
| path | what it is |
|---|---|
| `src/config.py` | every setting |
| `src/run_pipeline.py` | runs the steps and sweeps |
| `src/main_image_gen_SC_coast_local.py` | renders the images |
| `src/make_label_masks.py` | turns renders into labels |
| `src/train_hed_unet.py`, `hed_unet.py` | the model and training |
| `src/evaluate_hed_unet.py`, `predict_hed_unet.py` | scoring and looking at single images |
| `src/plot_runs.py` | graphs of runs |
| `src/functions.py` | rendering and orbit math (leave it alone unless you must) |

## References
- Heidler, K., et al. (2021). *HED-UNet: Combined Segmentation and Edge Detection for Monitoring the Antarctic Coastline.* Our model is adapted from https://github.com/khdlr/HED-UNet (MIT License).
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
- Cloud maps from shadedrelief.com; ephemerides from JPL HORIZONS.

Questions? Ask [Brendan Uribe]. ->discord
