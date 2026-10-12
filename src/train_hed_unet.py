import os # run folder
import sys # command line options
import csv # loss history
import json # config snapshot
import time # timer
import torch
import torch.nn.functional as F # cross entropy, pooling, log sigmoid
import torch.optim as optim # optimizers
from torch.utils.data import DataLoader # takes dataset and feeds to model batches
from coastline_dataset import CoastlineLabelDataset, get_splits # rgb images + multi-class label maps
from hed_unet import HEDUNet # imports HED-UNet model
import config as C # all settings (resolution, epochs, paths ...) live in src/config.py

# classes in the label maps (from make_label_masks.py): 0 space, 1 water, 2 land, 3 cloud, 4 dark
# model output channels: 0 to NUM_CLASSES-1 = class scores, last = coastline edge
SPACE, WATER, LAND, DARK = 0, 1, 2, 4


def prepare_labels(label): # apply C.MERGE_DARK to a label tensor or numpy array (returns a copy)
    label = label.clone() if torch.is_tensor(label) else label.copy()
    if C.MERGE_DARK:
        label[label == DARK] = SPACE
    return label


def coastline_from_label(label): # label [B,H,W] -> coastline [B,1,H,W] of 0/1
    # coastline = land pixel touching water OR water pixel touching land
    # cloud/dark/space boundaries are never coastline
    land = (label == LAND).float().unsqueeze(1)
    water = (label == WATER).float().unsqueeze(1)
    near_land = F.max_pool2d(land, 3, stride=1, padding=1) # 1 if any land in 3x3 neighborhood
    near_water = F.max_pool2d(water, 3, stride=1, padding=1)
    return ((land * near_water) + (water * near_land)).clamp(max=1)


def get_targets(label): # builds (class map, coastline) targets at every scale the model predicts at
    with torch.no_grad():
        edge = coastline_from_label(label)
        targets = []
        for level in range(C.STACK_HEIGHT + 1):
            s = 2 ** level
            level_label = label[:, ::s, ::s] # nearest downsample keeps class numbers
            level_edge = F.max_pool2d(edge, s) if s > 1 else edge # keep thin coastline visible at coarse levels
            targets.append((level_label, level_edge))
    return targets # full size first, same order as model's level outputs


def coastline_bce(logits, target): # BCE that boosts rare coastline pixels, capped at C.MAX_COASTLINE_WEIGHT
    # coastline is ~1-2% of pixels, so missed coastline pixels are weighted up to the cap x more
    # (full balancing = ~65x made the model mark almost the whole Earth as coastline, cap keeps it thin)
    with torch.no_grad():
        beta = target.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6) # fraction of coastline pixels per image
        pos_weight = ((1 - beta) / beta).clamp(max=C.MAX_COASTLINE_WEIGHT)
    return (-pos_weight * target * F.logsigmoid(logits)
            - (1 - target) * F.logsigmoid(-logits)).mean()


def hed_loss(prediction, target): # class loss + coastline loss for one scale
    label, edge = target
    class_loss = F.cross_entropy(prediction[:, :C.NUM_CLASSES], label)
    edge_loss = coastline_bce(prediction[:, C.NUM_CLASSES:], edge)
    return class_loss + edge_loss


def batch_loss(model, images, labels): # merged output + every level (deep supervision), same for training and validation
    combined, levels = model(images)
    targets = get_targets(labels)
    loss = hed_loss(combined, targets[0])
    for level_pred, level_target in zip(levels, targets):
        loss = loss + hed_loss(level_pred, level_target)
    return loss


def build_model(): # same model for training, prediction and evaluation
    return HEDUNet(in_channels=3, out_channels=C.NUM_CLASSES + 1,
                   base_channels=C.BASE_CHANNELS, stack_height=C.STACK_HEIGHT)


def train(): # trains with the current settings in config.py, saves everything to C.RUN_DIR, returns a summary dict
    os.makedirs(C.RUN_DIR, exist_ok=True)
    with open(f"{C.RUN_DIR}/config.json", "w") as f:
        json.dump(C.as_dict(), f, indent=2)

    torch.manual_seed(C.SEED)

    # Dataset - rgb images + label maps (validation images are never in here)
    (image_dir, label_dir, train_files), (val_image_dir, val_label_dir, val_files) = get_splits()
    dataset = CoastlineLabelDataset(image_dir, label_dir, img_size=C.IMG_SIZE, files=train_files)
    # shuffle for randomly mixed batches each epoch
    loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = None # validation loss is only logged (never trained on), so overfitting shows up in the loss curve
    if val_files:
        val_dataset = CoastlineLabelDataset(val_image_dir, val_label_dir, img_size=C.IMG_SIZE, files=val_files)
        val_loader = DataLoader(val_dataset, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Run '{C.RUN_ID}': {len(dataset)} training images, {len(val_files)} validation images, "
          f"IMG_SIZE {C.IMG_SIZE}, {C.NUM_CLASSES} classes")

    # Model - cuda for nvidia not amd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=C.LEARNING_RATE)

    history = [] # (epoch, train loss, validation loss, seconds)
    # early stopping: needs validation loss; keeps the model from the epoch with the lowest validation loss
    early_stopping = C.PATIENCE > 0 and val_loader is not None
    best_val, best_epoch, epochs_since_best, stopped_early = float("inf"), 0, 0, False
    total_start_time = time.time()

    for epoch in range(C.EPOCHS): # training loop for each epoch
        model.train()
        epoch_start_time = time.time() # timer for each epoch
        total_loss = 0 # loss initialize

        for images, labels in loader:
            images = images.to(device)
            labels = prepare_labels(labels).to(device)

            # loss = final merged output + every level's output (deep supervision)
            loss = batch_loss(model, images, labels)

            optimizer.zero_grad() # clearing old gradient info before new updates
            loss.backward() # calculate each model weight contribution to error (math step)
            optimizer.step() # updates model weight (learning)

            total_loss += loss.item() # accumulate loss for epoch

        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(loader) # avg loss across batches

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = sum(batch_loss(model, im.to(device), prepare_labels(lb).to(device)).item()
                               for im, lb in val_loader) / len(val_loader)
        history.append((epoch + 1, avg_loss, val_loss, epoch_time))
        print(f"Epoch {epoch+1}/{C.EPOCHS}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f} sec")

        if C.CHECKPOINT_EVERY and (epoch + 1) % C.CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), f"{C.RUN_DIR}/checkpoint_epoch_{epoch+1}.pth")

        if early_stopping:
            if val_loss < best_val: # new best (NaN never counts as better): this is the model that gets kept
                best_val, best_epoch, epochs_since_best = val_loss, epoch + 1, 0
                torch.save(model.state_dict(), C.MODEL_PATH)
            else:
                epochs_since_best += 1
                if epochs_since_best >= C.PATIENCE:
                    stopped_early = True
                    print(f"Early stop: validation loss has not improved for {C.PATIENCE} epochs "
                          f"(best was epoch {best_epoch}, val loss {best_val:.4f})")
                    break

    if not early_stopping or best_epoch == 0: # no early stopping (or no usable validation loss): keep the last epoch
        torch.save(model.state_dict(), C.MODEL_PATH)
    total_time = time.time() - total_start_time
    with open(f"{C.RUN_DIR}/train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "val_loss", "seconds"])
        writer.writerows(history)
    kept = f"best epoch {best_epoch} of {len(history)} run" if early_stopping and best_epoch else f"last epoch ({len(history)})"
    print(f"Model saved to {C.MODEL_PATH} ({kept}; training took {total_time/60:.2f} minutes)")
    try: # loss curve picture in the run folder
        from plot_runs import plot_run_loss
        plot_run_loss(C.RUN_DIR)
    except Exception as e: # a plotting problem must never lose a finished training run
        print(f"(could not save loss curve: {e})")
    if not (early_stopping and best_epoch): # not tracked above: take the lowest validation loss from the log
        valid = [(v, e) for e, _, v, _ in history if v == v] # v == v skips NaN
        best_val, best_epoch = min(valid) if valid else (float("nan"), 0)
    return dict(train_images=len(dataset), val_images=len(val_files),
                final_loss=history[-1][1] if history else float("nan"),
                final_val_loss=history[-1][2] if history else float("nan"),
                best_val_loss=best_val, best_val_epoch=best_epoch, epochs_run=len(history),
                stopped_early=stopped_early, train_minutes=total_time / 60)


if __name__ == "__main__":
    C.apply_cli(sys.argv[1:]) # e.g. --set IMG_SIZE=256 --set EPOCHS=50
    train()
