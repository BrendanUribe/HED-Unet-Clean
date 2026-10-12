# scores the trained HED-UNet (C.MODEL_PATH, the run chosen in config.py / --run) on the VALIDATION set
# prints averages over all images, saves one overlay plot per image + summary.csv + metrics.json in the run folder
# make the validation set with:  python src/make_label_masks.py validation   (or VAL_SOURCE = "split" in config.py)

import os # file finding/paths
import sys # command line options
import csv # summary table
import json # metrics
import time # inference timing
import numpy as np # math
import torch
import matplotlib
matplotlib.use("Agg") # save plots to files, no windows
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF # image to tensor
from scipy.spatial import cKDTree # nearest-point distances for coastline error
from coastline_dataset import letterbox, get_splits # same no-stretch resize + padding as training
from coastline_lines import predicted_line, true_boundary_points # thin lines + true coastline
from train_hed_unet import build_model, prepare_labels # same model + label handling as training
import config as C # all settings live in src/config.py


def load_model(device):
    model = build_model().to(device)
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))
    model.eval() # evaluation mode for predicting not training
    return model


def evaluate(save_plots=True): # returns a flat dict of metrics over all validation images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    image_dir, label_dir, image_files = get_splits()[1]
    print(f"Evaluating {len(image_files)} validation images with {C.MODEL_PATH} (IMG_SIZE {C.IMG_SIZE})\n")
    if save_plots:
        os.makedirs(C.RESULTS_DIR, exist_ok=True)

    # totals over all images
    intersections = np.zeros(C.NUM_CLASSES)
    unions = np.zeros(C.NUM_CLASSES)
    correct_pixels = total_pixels = 0
    all_errors = [] # sub-pixel line point -> nearest true coastline, original pixels
    all_found = [] # per true coastline point: line point within 1 model pixel?
    all_found_fixed = [] # per true coastline point: line point within C.FOUND_TOLERANCE_PX ORIGINAL pixels (same at any IMG_SIZE)
    full_inter, full_union = np.zeros(C.NUM_CLASSES), np.zeros(C.NUM_CLASSES) # class scores at ORIGINAL resolution
    full_correct = full_total = 0
    infer_seconds = [] # model forward pass per image
    images_missed = images_false_alarm = 0 # true coastline but no predicted line / predicted line but no true coastline
    rows = []

    for img_name in image_files:
        number = img_name[len("earth_img_"):-len(".png")]
        image_full = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        label_full = Image.open(os.path.join(label_dir, f"earth_img_LABEL{number}.png"))

        image_small, (scale, pad_x, pad_y) = letterbox(image_full, C.IMG_SIZE, Image.BILINEAR)
        label_small, _ = letterbox(label_full, C.IMG_SIZE, Image.NEAREST)
        truth = prepare_labels(np.array(label_small))
        truth_full = prepare_labels(np.array(label_full))

        batch = TF.to_tensor(image_small).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            combined, _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_seconds.append(time.perf_counter() - start)
            class_map = combined[0, :C.NUM_CLASSES].argmax(dim=0).cpu().numpy()
            edge_prob = torch.sigmoid(combined[0, C.NUM_CLASSES]).cpu().numpy()
        edge_band = (edge_prob > 0.5).astype(np.uint8)

        # class scores on the real image area only (not the letterbox padding)
        valid = np.zeros_like(truth, dtype=bool)
        new_w, new_h = round(image_full.width / scale), round(image_full.height / scale)
        valid[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = True
        for c in range(C.NUM_CLASSES):
            pred_c, truth_c = (class_map == c) & valid, (truth == c) & valid
            intersections[c] += np.logical_and(pred_c, truth_c).sum()
            unions[c] += np.logical_or(pred_c, truth_c).sum()
        # same scores at ORIGINAL resolution: crop away the padding, scale the class map back up, compare with the full label
        class_full = np.array(Image.fromarray(class_map[pad_y:pad_y + new_h, pad_x:pad_x + new_w].astype(np.uint8))
                              .resize(image_full.size, Image.NEAREST))
        for c in range(C.NUM_CLASSES):
            pred_c, truth_c = class_full == c, truth_full == c
            full_inter[c] += np.logical_and(pred_c, truth_c).sum()
            full_union[c] += np.logical_or(pred_c, truth_c).sum()
        full_correct += (class_full == truth_full).sum()
        full_total += truth_full.size
        image_correct = ((class_map == truth) & valid).sum()
        correct_pixels += image_correct
        total_pixels += valid.sum()

        # coastline: thin sub-pixel line vs exact true boundary, in original pixels
        segments, line_full, _ = predicted_line(edge_band, edge_prob, scale, pad_x, pad_y)
        true_boundary = true_boundary_points(truth_full)
        row = dict(image=img_name, pixel_accuracy=image_correct / valid.sum(), true_coast_points=len(true_boundary),
                   line_points=len(line_full), median_error_px="", median_error_mrad="", coast_found_pct="",
                   coast_found_fixed_pct="")
        if len(true_boundary) and len(line_full):
            errors = cKDTree(true_boundary).query(line_full)[0]
            dist_to_line = cKDTree(line_full).query(true_boundary)[0]
            found = dist_to_line <= scale
            all_errors.append(errors)
            all_found.append(found)
            all_found_fixed.append(dist_to_line <= C.FOUND_TOLERANCE_PX)
            row.update(median_error_px=np.median(errors), median_error_mrad=np.median(errors) * C.MRAD_PER_PIXEL,
                       coast_found_pct=100 * found.mean(), coast_found_fixed_pct=100 * (dist_to_line <= C.FOUND_TOLERANCE_PX).mean())
            print(f"{img_name}: pixel acc {row['pixel_accuracy']:.3f}, coastline error median {np.median(errors):.2f} px, "
                  f"found {100 * found.mean():.0f}%")
        elif len(true_boundary): # coastline is there but the model drew none: every true point counts as NOT found
            images_missed += 1
            all_found.append(np.zeros(len(true_boundary), dtype=bool))
            all_found_fixed.append(np.zeros(len(true_boundary), dtype=bool))
            row.update(coast_found_pct=0.0, coast_found_fixed_pct=0.0)
            print(f"{img_name}: pixel acc {row['pixel_accuracy']:.3f}, coastline MISSED (true coastline, none predicted)")
        elif len(line_full): # no coastline in the image but the model drew one
            images_false_alarm += 1
            print(f"{img_name}: pixel acc {row['pixel_accuracy']:.3f}, FALSE coastline predicted (none in the image)")
        else:
            print(f"{img_name}: pixel acc {row['pixel_accuracy']:.3f}, no coastline (true or predicted)")
        rows.append(row)

        if save_plots: # overlay plot saved per image: true coastline, predicted band, predicted thin line
            ys, xs = np.nonzero(edge_band)
            plt.figure(figsize=(12, 9))
            plt.imshow(np.array(image_full))
            plt.scatter((xs - pad_x + 0.5) * scale - 0.5, (ys - pad_y + 0.5) * scale - 0.5, s=4, c="red", marker="s",
                        linewidths=0, alpha=0.5, label="predicted band (model pixels)")
            if len(true_boundary):
                plt.scatter(true_boundary[:, 0], true_boundary[:, 1], s=0.3, c="lime", linewidths=0, label="true coastline")
            for i, seg in enumerate(segments):
                plt.plot(seg[:, 0], seg[:, 1], c="cyan", linewidth=1,
                         label="predicted line (thinned, sub-pixel)" if i == 0 else None)
            plt.title(f"{img_name} - validation overlay")
            plt.legend(loc="lower right", markerscale=4)
            plt.axis("off")
            plt.savefig(os.path.join(C.RESULTS_DIR, f"earth_img_{number}_overlay.png"), dpi=100, bbox_inches="tight")
            plt.close()

    # averages over ALL validation images
    iou = np.where(unions > 0, intersections / np.maximum(unions, 1), np.nan)
    metrics = dict(val_images=len(image_files),
                   pixel_accuracy=correct_pixels / total_pixels if total_pixels else float("nan"),
                   mean_iou=float(np.nanmean(iou)) if np.any(unions > 0) else float("nan"))
    for name, value in zip(C.CLASS_NAMES, iou):
        metrics[f"iou_{name.replace('/', '_')}"] = float(value)
    # the same scores at ORIGINAL resolution: comparable between different IMG_SIZE runs
    full_iou = np.where(full_union > 0, full_inter / np.maximum(full_union, 1), np.nan)
    metrics["full_pixel_accuracy"] = full_correct / full_total if full_total else float("nan")
    metrics["full_mean_iou"] = float(np.nanmean(full_iou)) if np.any(full_union > 0) else float("nan")
    for name, value in zip(C.CLASS_NAMES, full_iou):
        metrics[f"full_iou_{name.replace('/', '_')}"] = float(value)
    metrics["images_missed"] = images_missed # true coastline but the model drew none
    metrics["images_false_alarm"] = images_false_alarm # coastline drawn where there is none
    metrics["n_params"] = int(n_params)
    metrics["infer_ms_median"] = float(np.median(infer_seconds) * 1000) if infer_seconds else float("nan")

    print(f"\n===== VALIDATION RESULTS ({len(image_files)} images) =====")
    print("IoU per class:")
    for name, value in zip(C.CLASS_NAMES, iou):
        print(f"  {name:10s}: {value:.4f}" if not np.isnan(value) else f"  {name:10s}: not in any image")
    print(f"Pixel accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"At original resolution: pixel accuracy {metrics['full_pixel_accuracy']:.4f}, mean IoU {metrics['full_mean_iou']:.4f}")
    print(f"Model: {n_params:,} parameters, {metrics['infer_ms_median']:.1f} ms per image (median forward pass)")

    if all_errors:
        errors = np.concatenate(all_errors)
        metrics.update(coast_error_median_px=float(np.median(errors)), coast_error_mean_px=float(errors.mean()),
                       coast_error_p90_px=float(np.percentile(errors, 90)),
                       coast_error_median_mrad=float(np.median(errors) * C.MRAD_PER_PIXEL),
                       coast_error_p90_mrad=float(np.percentile(errors, 90) * C.MRAD_PER_PIXEL))
        print(f"\nCoastline error (predicted line points vs true coastline, pooled over all images, "
              f"1 original pixel = {C.MRAD_PER_PIXEL:.3f} mrad):")
        print(f"  error median {np.median(errors):.2f} px ({np.median(errors) * C.MRAD_PER_PIXEL:.3f} mrad), "
              f"mean {errors.mean():.2f} px ({errors.mean() * C.MRAD_PER_PIXEL:.3f} mrad), "
              f"90% under {np.percentile(errors, 90):.2f} px ({np.percentile(errors, 90) * C.MRAD_PER_PIXEL:.3f} mrad)")
    if all_found: # includes images where a true coastline existed but none was predicted (counted as not found)
        found, found_fixed = np.concatenate(all_found), np.concatenate(all_found_fixed)
        metrics.update(coast_found_pct=float(100 * found.mean()), coast_found_fixed_pct=float(100 * found_fixed.mean()))
        print(f"  true coastline found (line within 1 model pixel): {100 * found.mean():.0f}%")
        print(f"  true coastline found (line within {C.FOUND_TOLERANCE_PX} original pixels): {100 * found_fixed.mean():.0f}%")
    if images_missed or images_false_alarm:
        print(f"  images with true coastline but none predicted: {images_missed}, "
              f"coastline predicted where there is none: {images_false_alarm}")
    if not all_errors and not all_found:
        print("\nNo coastline in any validation image")

    os.makedirs(C.RUN_DIR, exist_ok=True)
    with open(os.path.join(C.RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if save_plots:
        with open(os.path.join(C.RESULTS_DIR, "summary.csv"), "w", newline="") as f: # one row per image
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved overlay plots + summary.csv to {C.RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    C.apply_cli(sys.argv[1:], from_run=True) # e.g. --run <run folder name>
    evaluate()
