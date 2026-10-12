import os #file finding/paths
import random # reproducible train/validation split
import numpy as np # label arrays
import torch # label tensors
from PIL import Image # open images
from torch.utils.data import Dataset # pytorch dataset class - can feed images into nn
import torchvision.transforms as T # image transformations
import config as C # all settings


def letterbox(img, size, resample): # shrink keeping shape (no stretching), then pad to a size x size square with black
    # returns padded image + (scale, pad_x, pad_y) so pixels can be mapped back to the original image:
    #   x_original = (x_small - pad_x + 0.5) * scale - 0.5, same for y
    w, h = img.size
    scale = max(w, h) / size # original pixels per small pixel, same in x and y
    new_w, new_h = round(w / scale), round(h / scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = Image.new(img.mode, (size, size), 0) # 0 = black = space class
    canvas.paste(img.resize((new_w, new_h), resample), (pad_x, pad_y))
    return canvas, (scale, pad_x, pad_y)


def list_images(image_dir): # only plain images earth_img_<N>.png, skips MASK/LABEL/render files if they share the folder
    return sorted(
        f for f in os.listdir(image_dir)
        if f.startswith("earth_img_") and f.endswith(".png")
        and f[len("earth_img_"):-len(".png")].replace("_", "").isdigit()
    )


def get_splits(): # -> (train, val), each (image_dir, label_dir, [image files]) following C.VAL_SOURCE
    if C.VAL_SOURCE == "folder":
        train = (C.TRAIN_IMAGES, C.TRAIN_LABELS, list_images(C.TRAIN_IMAGES))
        val = (C.VAL_IMAGES, C.VAL_LABELS, list_images(C.VAL_IMAGES))
    else:
        files = list_images(C.TRAIN_IMAGES) # one pool, hold out VAL_FRACTION of it
        shuffled = files[:]
        random.Random(C.SPLIT_SEED).shuffle(shuffled) # same seed = same split every run
        n_val = min(max(1, round(len(files) * C.VAL_FRACTION)), len(files) - 1)
        train = (C.TRAIN_IMAGES, C.TRAIN_LABELS, sorted(shuffled[n_val:]))
        val = (C.TRAIN_IMAGES, C.TRAIN_LABELS, sorted(shuffled[:n_val]))
    if C.TRAIN_FRACTION < 1: # use only part of the training images, validation is untouched
        keep = train[2][:]
        random.Random(C.SPLIT_SEED).shuffle(keep)
        keep = sorted(keep[:max(1, round(len(keep) * C.TRAIN_FRACTION))])
        train = (train[0], train[1], keep)
    return train, val


# pairs earth_img_<N>.png with earth_img_LABEL<N>.png
# label values: 0 space, 1 water, 2 land, 3 cloud, 4 dark (made by make_label_masks.py)
# images are letterboxed (shrunk without stretching + padded with space) so geometry stays correct
class CoastlineLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=256, files=None): # img_size = side of the square fed to the model
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_files = files if files is not None else list_images(image_dir) # files = subset from get_splits()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        number = img_name.replace("earth_img_", "").replace(".png", "")

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, f"earth_img_LABEL{number}.png")) # keep raw class numbers, no convert

        image, _ = letterbox(image, self.img_size, Image.BILINEAR)
        label, _ = letterbox(label, self.img_size, Image.NEAREST) # nearest so class numbers never get blended

        image = T.functional.to_tensor(image)
        label = torch.from_numpy(np.array(label, dtype=np.int64)) # [H, W] class numbers

        return image, label
