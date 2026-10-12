# run the whole workflow (or any part of it) from one command, and sweep settings to compare runs
#
#   python src/run_pipeline.py                                   train + evaluate with the settings in src/config.py
#   python src/run_pipeline.py --stages generate,labels,val-labels,train,evaluate    everything from scratch
#   python src/run_pipeline.py --set IMG_SIZE=256 --set EPOCHS=50                   one run with overrides
#   python src/run_pipeline.py --sweep IMG_SIZE=256,512 --sweep STACK_HEIGHT=4,5    4 runs (every combination)
#   python src/run_pipeline.py --sweep TRAIN_FRACTION=0.25,0.5,0.75,1.0      learning curve: same validation set, less training data
#
# generate / labels / val-labels run ONCE at the start, even in a sweep, so every run trains on the same images
# stages (in this order): generate  render images with POV-Ray (needs SPICE kernels, maps, POV-Ray; writes to RENDER_DIR)
#                         labels    renders -> dataset/images + dataset/labels, archives the used renders
#                         val-labels renders -> dataset/validation_*, archives the used renders
#                         train     trains, saves runs/<run>/model.pth
#                         evaluate  scores the validation set, saves runs/<run>/metrics.json + overlay plots
# options: --reuse  skip training when runs/<run>/model.pth already exists
#          --no-plots  skip the per-image overlay plots during evaluation (faster sweeps)
# every run is appended to runs/results.csv so runs can be compared side by side

import os
import sys
import csv
import json
import itertools # every combination of sweep values
import subprocess # runs the image generator
import config as C

STAGES = ["generate", "labels", "val-labels", "train", "evaluate"]
DATA_STAGES = ["generate", "labels", "val-labels"] # build the dataset: run ONCE, never per run of a sweep
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def append_results(row): # add one row to runs/results.csv, widening the columns if this row has new ones
    path = os.path.join(C.RUNS_DIR, "results.csv")
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("run_id") != row["run_id"]] + [row] # re-running a run replaces its row
    columns = ["run_id", "overrides"] + sorted({k for r in rows for k in r} - {"run_id", "overrides"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_data_stages(stages): # generate renders / build labels, once, with the current settings
    if "generate" in stages:
        print("Generating images (this can take a long time)...")
        subprocess.run([sys.executable, os.path.join("src", "main_image_gen_SC_coast_local.py")], cwd=REPO_ROOT, check=True)
    if "labels" in stages or "val-labels" in stages:
        from make_label_masks import build_labels
        if "labels" in stages:
            build_labels("train", archive=True)
        if "val-labels" in stages:
            build_labels("validation", archive=True)


def run_one(overrides, stages, reuse, plots): # one full run with these overrides, returns the results row
    C.apply(overrides)
    row = {"run_id": C.RUN_ID, "overrides": json.dumps({k: v for k, v in C._OVERRIDES.items()})}
    print(f"\n=================== RUN {C.RUN_ID} ===================")
    try:
        if "train" in stages:
            if reuse and os.path.exists(C.MODEL_PATH):
                print(f"--reuse: {C.MODEL_PATH} exists, skipping training")
            else:
                from train_hed_unet import train
                row.update(train())
        if "evaluate" in stages:
            from evaluate_hed_unet import evaluate
            row.update(evaluate(save_plots=plots))
    except Exception as e: # keep a sweep going if one combination fails (e.g. out of memory)
        print(f"RUN {C.RUN_ID} FAILED: {type(e).__name__}: {e}")
        row["error"] = f"{type(e).__name__}: {e}"
    append_results({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in row.items()})
    return row


def main(argv):
    stages, base, sweeps, reuse, plots, i = ["train", "evaluate"], {}, {}, False, True, 0
    while i < len(argv):
        a = argv[i]
        if a == "--stages":
            stages = argv[i + 1].split(",")
            unknown = [s for s in stages if s not in STAGES]
            if unknown:
                sys.exit(f"unknown stage {unknown}, valid: {', '.join(STAGES)}")
            i += 2
        elif a == "--set":
            key, _, value = argv[i + 1].partition("=")
            base[key] = value
            i += 2
        elif a == "--run":
            base["RUN_NAME"] = argv[i + 1]
            i += 2
        elif a == "--sweep":
            key, _, values = argv[i + 1].partition("=")
            sweeps[key] = values.split(",")
            i += 2
        elif a == "--reuse":
            reuse, i = True, i + 1
        elif a == "--no-plots":
            plots, i = False, i + 1
        else:
            sys.exit(f"unknown option '{a}', see the top of src/run_pipeline.py")

    if sweeps and "RUN_NAME" in base:
        sys.exit("--run names a single run, don't combine it with --sweep (runs are named from their settings)")
    combos = [dict(zip(sweeps, values)) for values in itertools.product(*sweeps.values())] or [{}]
    C.apply({**base, **combos[0]}) # validates every combination before anything starts
    for combo in combos:
        C.apply({**base, **combo})
    data_stages = [s for s in STAGES if s in stages and s in DATA_STAGES]
    run_stages = [s for s in stages if s not in DATA_STAGES]
    print(f"{len(combos)} run(s), stages: {', '.join(stages)}")

    if data_stages: # the dataset is built once, before the first run, so every run of a sweep trains on identical images
        C.apply(base)
        print(f"\n=================== DATASET ({', '.join(data_stages)}, once for all runs) ===================")
        run_data_stages(data_stages) # a failure here stops everything: there is nothing valid to train on
    if not run_stages:
        return

    rows = [run_one({**base, **combo}, run_stages, reuse, plots) for combo in combos]
    if len(rows) > 1 or "evaluate" in run_stages:
        shown = ["run_id", "coast_error_median_px", "coast_error_p90_px", "full_mean_iou", "coast_found_fixed_pct",
                 "best_val_loss", "train_minutes"]
        print("\n=================== SUMMARY ===================")
        for r in rows:
            print("  ".join(f"{k}={r[k]:.4g}" if isinstance(r.get(k), float) else f"{k}={r.get(k, '-')}" for k in shown)
                  + (f"  ERROR {r['error']}" if "error" in r else ""))
        print(f"\nAll runs: {os.path.join(C.RUNS_DIR, 'results.csv')}")


if __name__ == "__main__":
    main(sys.argv[1:])
