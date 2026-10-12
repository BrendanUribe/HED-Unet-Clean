# ONE place for every setting. All scripts do `import config as C` and read C.<NAME> at run time.
# Edit the defaults below, or override a few without touching the file:
#   python src/run_pipeline.py --set IMG_SIZE=256 --set STACK_HEIGHT=4
# Each run gets its own folder runs/<RUN_ID>/ (model, checkpoints, config snapshot, metrics, plots),
# so experiments never overwrite each other and the folder name says what was changed.

import os
import re
import json
import zlib # short hash for very long run names

# ============================ EDIT THESE ============================

# ---- resolution / model ----
IMG_SIZE = 512            # square side fed to the model (long image side shrinks to this)
STACK_HEIGHT = 5          # number of down/up levels, IMG_SIZE must be divisible by 2**STACK_HEIGHT
BASE_CHANNELS = 16        # feature maps at the first level (doubles each level down)

# ---- classes ----
# label files hold 5 classes: 0 space, 1 water, 2 land, 3 cloud, 4 dark (night side)
# MERGE_DARK = True treats dark as space: both are pure black in the images, so the model can only tell
# them apart by the disk shape, which failed at 512 res. False = 5 classes (retrain after changing)
MERGE_DARK = True

# ---- training ----
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
MAX_COASTLINE_WEIGHT = 10.0   # cap on the extra weight for rare coastline pixels (full balancing ~65x was too much)
CHECKPOINT_EVERY = 10         # save a checkpoint every N epochs (0 = never)
SEED = 0                      # model init + batch shuffling
# early stopping: stop when validation loss has not improved for PATIENCE epochs and keep the model from the best epoch.
# EPOCHS is then only the ceiling. 0 = off (always trains EPOCHS and keeps the last epoch)
PATIENCE = 10

# ---- train / validation split ----
# "folder": validation images come from their own folder (make_label_masks.py validation)
# "split" : one pool of labelled images, VAL_FRACTION of it is held out for validation (same SPLIT_SEED = same split)
VAL_SOURCE = "folder"
VAL_FRACTION = 0.2
SPLIT_SEED = 0
# learning-curve knob: train on only this share of the training images (validation set stays the SAME, so runs are comparable)
TRAIN_FRACTION = 1.0

# ---- paths ----
DATA_DIR = "dataset"          # dataset/images, dataset/labels, dataset/validation_images, dataset/validation_labels
RENDER_DIR = "."              # where the generator saves earth_img_LANDFLAT<N>.png etc (its working directory)
RUNS_DIR = "runs"             # one subfolder per run
RUN_NAME = None               # None = name built from whatever you overrode (or "default"), or set your own

# ---- evaluation ----
FOUND_TOLERANCE_PX = 2        # a true coastline point counts as "found" if a predicted point is within this many ORIGINAL pixels

# ---- camera (case_type 3000 in functions.py): turns pixels into milliradians ----
FOCAL_LEN_MM = 35
PIXEL_SIZE_MM = 4.96e-3

# ====================================================================

_DEFAULTS = {k: v for k, v in list(globals().items()) if k.isupper()}
_OVERRIDES = {}
_NOT_IN_NAME = {"RUN_NAME", "DATA_DIR", "RENDER_DIR", "RUNS_DIR"} # settings that do not change the model/results

# filled in by _derive(), do not edit:
# CLASS_NAMES NUM_CLASSES MRAD_PER_PIXEL RUN_ID RUN_DIR MODEL_PATH
# TRAIN_IMAGES TRAIN_LABELS VAL_IMAGES VAL_LABELS (+ *_PREVIEWS) RESULTS_DIR


_SHORT = {"IMG_SIZE": "img", "STACK_HEIGHT": "sh", "BASE_CHANNELS": "bc", "BATCH_SIZE": "bs", "LEARNING_RATE": "lr",
          "TRAIN_FRACTION": "tf", "EPOCHS": "ep", "SEED": "seed", "VAL_FRACTION": "vf", "VAL_SOURCE": "val-",
          "SPLIT_SEED": "ssd", "PATIENCE": "pat", "MERGE_DARK": "merge", "MAX_COASTLINE_WEIGHT": "cw", "CHECKPOINT_EVERY": "ck"} # run name pieces


def _short_value(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _coerce(value, default):  # "256" -> 256 etc, based on the type of the default
    if not isinstance(value, str) or default is None or isinstance(default, str):
        return value
    if isinstance(default, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected true/false, got '{value}'")
    return type(default)(value)


def _derive():
    g = globals()
    if g["IMG_SIZE"] % (2 ** g["STACK_HEIGHT"]) != 0:
        raise ValueError(f"IMG_SIZE {g['IMG_SIZE']} must be divisible by 2**STACK_HEIGHT = {2 ** g['STACK_HEIGHT']}")
    if g["VAL_SOURCE"] not in ("folder", "split"):
        raise ValueError("VAL_SOURCE must be 'folder' or 'split'")
    if not 0 < g["VAL_FRACTION"] < 1:
        raise ValueError("VAL_FRACTION must be between 0 and 1")
    if not 0 < g["TRAIN_FRACTION"] <= 1:
        raise ValueError("TRAIN_FRACTION must be in (0, 1]")

    g["CLASS_NAMES"] = ["space/dark", "water", "land", "cloud"] if g["MERGE_DARK"] \
        else ["space", "water", "land", "cloud", "dark"]
    g["NUM_CLASSES"] = len(g["CLASS_NAMES"])
    g["MRAD_PER_PIXEL"] = g["PIXEL_SIZE_MM"] / g["FOCAL_LEN_MM"] * 1000  # angle covered by one ORIGINAL image pixel

    changed = {k: v for k, v in sorted(_OVERRIDES.items()) if k not in _NOT_IN_NAME} # sorted: same name whatever the flag order
    auto = "_".join(f"{_SHORT.get(k, k.lower())}{_short_value(v)}" for k, v in changed.items()) or "default"
    auto = re.sub(r"[^A-Za-z0-9._=-]+", "-", auto)
    if len(auto) > 100: # keep folder names short, config.json in the run folder has the full list
        auto = auto[:90] + "_" + format(zlib.crc32(auto.encode()), "08x")
    g["RUN_ID"] = g["RUN_NAME"] or auto
    g["RUN_DIR"] = f"{g['RUNS_DIR']}/{g['RUN_ID']}"
    g["MODEL_PATH"] = f"{g['RUN_DIR']}/model.pth"
    g["RESULTS_DIR"] = f"{g['RUN_DIR']}/validation_results"

    d = g["DATA_DIR"]
    g["TRAIN_IMAGES"], g["TRAIN_LABELS"], g["TRAIN_PREVIEWS"] = f"{d}/images", f"{d}/labels", f"{d}/label_previews"
    g["VAL_IMAGES"], g["VAL_LABELS"], g["VAL_PREVIEWS"] = \
        f"{d}/validation_images", f"{d}/validation_labels", f"{d}/validation_label_previews"


def apply(overrides=None):  # reset to the defaults above, then apply {NAME: value} on top (values may be strings)
    g = globals()
    g.update(_DEFAULTS)
    _OVERRIDES.clear()
    for key, value in (overrides or {}).items():
        key = key.upper()
        if key not in _DEFAULTS:
            raise KeyError(f"unknown setting '{key}', valid: {', '.join(_DEFAULTS)}")
        g[key] = _OVERRIDES[key] = _coerce(value, _DEFAULTS[key])
    _derive()


def apply_cli(argv, from_run=False):  # pulls `--set KEY=VALUE` (repeatable) and `--run NAME` out of argv, applies them, returns the rest
    # from_run=True (predict / evaluate): start from the settings the named run was TRAINED with (runs/<name>/config.json),
    # so the model shape, IMG_SIZE, classes and split match its weights; explicit --set values still win
    overrides, rest, i = {}, [], 0
    while i < len(argv):
        if argv[i] == "--set" and i + 1 < len(argv):
            key, _, value = argv[i + 1].partition("=")
            overrides[key.upper()] = value
            i += 2
        elif argv[i] == "--run" and i + 1 < len(argv):
            overrides["RUN_NAME"] = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if from_run and overrides.get("RUN_NAME"):
        apply({k: v for k, v in overrides.items() if k in ("RUNS_DIR", "RUN_NAME")})  # find the run folder
        path = f"{RUN_DIR}/config.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"no saved run at {path}, train it first or check the name (runs are in {RUNS_DIR}/)")
        with open(path) as f:
            saved = {k: v for k, v in json.load(f).items() if k in _DEFAULTS}
        overrides = {**saved, **overrides}
    apply(overrides)
    return rest


def as_dict():  # every setting (defaults + overrides + derived) for saving next to a run
    g = globals()
    keys = list(_DEFAULTS) + ["CLASS_NAMES", "NUM_CLASSES", "MRAD_PER_PIXEL", "RUN_ID", "RUN_DIR", "MODEL_PATH"]
    return {k: g[k] for k in keys}


apply()
