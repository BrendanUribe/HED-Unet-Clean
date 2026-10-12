# combines the 4 flat renders from gen_earth_label_renders (functions.py) into ONE label map per image
# label map = single channel png, each pixel value is a class number (not a color):
#   0 = space, 1 = water, 2 = land, 3 = cloud, 4 = dark (night side)
# priority when classes overlap: space > dark > cloud > land/water
# coastline is NOT stored, training computes it from this map (land touching water only)
#
# run:  python src/make_label_masks.py                     -> training set   (dataset/images, dataset/labels)
#       python src/make_label_masks.py validation          -> validation set (dataset/validation_images, dataset/validation_labels)
#       add --archive to move the used renders to dataset/render_archive/<split>/ so the next batch starts clean
# it processes EVERY render set in C.RENDER_DIR, so archive/move old renders before generating a new set

import os # file finding/paths
import sys # command line option (train / validation)
import shutil # copy rgb images into the dataset folders
import numpy as np # math
from PIL import Image # open/save images
import config as C # folders live in src/config.py

# thresholds (0-1 brightness) - tune if labels look wrong in the previews
LAND_THRESHOLD = 0.08 # color_oceanblack map: ocean is black, anything brighter is land
CLOUD_THRESHOLD = 0.4 # cloud map coverage that counts as cloud, same as coverage_threshold in coastline_func
DARK_THRESHOLD = 0.1 # sunlit render: below this = night side / too dark to see coastline

SPACE, WATER, LAND, CLOUD, DARK = 0, 1, 2, 3, 4
PREVIEW_COLORS = np.array([
    [0, 0, 0], # space - black
    [30, 60, 200], # water - blue
    [40, 160, 60], # land - green
    [230, 230, 230], # cloud - white
    [90, 60, 110], # dark - purple
], dtype=np.uint8)


def load_gray(path): # render -> 0-1 brightness array (max over rgb so colored land counts fully)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).max(axis=2) / 255.0


def make_label(number, render_dir):
    land = load_gray(os.path.join(render_dir, f"earth_img_LANDFLAT{number}.png")) > LAND_THRESHOLD
    cloud = load_gray(os.path.join(render_dir, f"earth_img_CLOUDFLAT{number}.png")) > CLOUD_THRESHOLD
    disk = load_gray(os.path.join(render_dir, f"earth_img_DISK{number}.png")) > 0.5
    lit = load_gray(os.path.join(render_dir, f"earth_img_LIT{number}.png")) > DARK_THRESHOLD

    label = np.full(land.shape, WATER, dtype=np.uint8) # start as water, then overwrite in reverse priority
    label[land] = LAND
    label[cloud] = CLOUD
    label[~lit] = DARK
    label[~disk] = SPACE
    return label


def build_labels(split="train", archive=False): # returns (number labeled, list of skipped numbers)
    if split not in ("train", "validation"):
        raise ValueError(f"unknown split '{split}', use 'train' or 'validation'")
    render_dir = C.RENDER_DIR
    if split == "validation":
        image_dir, label_dir, preview_dir = C.VAL_IMAGES, C.VAL_LABELS, C.VAL_PREVIEWS
    else:
        image_dir, label_dir, preview_dir = C.TRAIN_IMAGES, C.TRAIN_LABELS, C.TRAIN_PREVIEWS # preview_dir None = skip
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    # find every image number that has a DISK render
    numbers = sorted(
        f[len("earth_img_DISK"):-len(".png")]
        for f in os.listdir(render_dir)
        if f.startswith("earth_img_DISK") and f.endswith(".png")
    )
    print(f"Found {len(numbers)} sets of renders in '{render_dir}' -> {split} set ({image_dir}, {label_dir})")

    skipped = [] # views with a missing render (usually a missing cloud map for that date, so POV-Ray failed)
    done = []
    for number in numbers:
        needed = [f"earth_img_{number}.png"] + [f"earth_img_{kind}{number}.png" for kind in ("LANDFLAT", "CLOUDFLAT", "DISK", "LIT")]
        missing = [f for f in needed if not os.path.exists(os.path.join(render_dir, f))]
        if missing:
            skipped.append(number)
            print(f"{number}: SKIPPED, missing {', '.join(missing)}")
            continue
        label = make_label(number, render_dir)
        Image.fromarray(label).save(os.path.join(label_dir, f"earth_img_LABEL{number}.png"))
        shutil.copy(os.path.join(render_dir, f"earth_img_{number}.png"), os.path.join(image_dir, f"earth_img_{number}.png"))
        if preview_dir:
            Image.fromarray(PREVIEW_COLORS[label]).save(os.path.join(preview_dir, f"earth_img_LABEL{number}.png"))
        done.append(number)

        fractions = np.bincount(label.ravel(), minlength=5) / label.size # class share of the image
        print(f"{number}: space {fractions[0]:.2f}, water {fractions[1]:.2f}, land {fractions[2]:.2f}, "
              f"cloud {fractions[3]:.2f}, dark {fractions[4]:.2f}")

    if archive and done: # move the used renders away so they are not picked up again
        archive_dir = os.path.join(C.DATA_DIR, "render_archive", split)
        os.makedirs(archive_dir, exist_ok=True)
        for number in done:
            for name in [f"earth_img_{number}.png"] + [f"earth_img_{kind}{number}.png" for kind in ("LANDFLAT", "CLOUDFLAT", "DISK", "LIT")]:
                shutil.move(os.path.join(render_dir, name), os.path.join(archive_dir, name))
        print(f"Moved {len(done)} render sets to {archive_dir}")

    print(f"Done: {len(done)} labeled, {len(skipped)} skipped")
    if skipped:
        print("Skipped views (render failed, usually a missing cloud map for that date):", ", ".join(skipped))
    return len(done), skipped


if __name__ == "__main__":
    args = C.apply_cli(sys.argv[1:])
    build_labels(split=next((a for a in args if not a.startswith("--")), "train"), archive="--archive" in args)
