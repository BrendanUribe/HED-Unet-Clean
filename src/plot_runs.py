# pictures for comparing runs (saved as PNG files, nothing pops up)
#
# every run already gets runs/<name>/loss_curve.png at the end of training (train loss solid, validation loss dashed)
#
#   python src/plot_runs.py                                        loss curves of ALL runs on one plot -> runs/loss_curves.png
#   python src/plot_runs.py --runs name1,name2                     only these runs
#   python src/plot_runs.py --metric coast_error_median_px --vs TRAIN_FRACTION
#                                                                  a metric against a swept setting (e.g. the learning curve)
#                                                                  -> runs/<metric>_vs_<setting>.png
#   python src/plot_runs.py --metric coast_error_median_px --vs IMG_SIZE --where STACK_HEIGHT=5 --where LEARNING_RATE=0.001
#                                                                  --where keeps only runs where that setting has that value
#                                                                  (anything not passed with --set counts as its default in config.py)

import os
import sys
import csv
import json
import matplotlib
matplotlib.use("Agg") # save files, no windows
import matplotlib.pyplot as plt
import config as C


def read_log(run_dir): # -> (epochs, train loss, validation loss) from train_log.csv
    epochs, train, val = [], [], []
    with open(os.path.join(run_dir, "train_log.csv"), newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train.append(float(row["avg_loss"]))
            val.append(float(row["val_loss"]) if row.get("val_loss") not in (None, "") else float("nan"))
    return epochs, train, val


def plot_run_loss(run_dir): # runs/<name>/loss_curve.png
    epochs, train, val = read_log(run_dir)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train, label="train loss")
    if any(v == v for v in val): # skip when there was no validation set (NaN)
        plt.plot(epochs, val, "--", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title(os.path.basename(run_dir.rstrip("/")), fontsize=8)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=110, bbox_inches="tight")
    plt.close()


def plot_all_losses(run_names): # every run on one plot: solid = train, dashed = validation, same color per run
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(run_names):
        run_dir = os.path.join(C.RUNS_DIR, name)
        if not os.path.exists(os.path.join(run_dir, "train_log.csv")):
            print(f"skipping {name}: no train_log.csv")
            continue
        epochs, train, val = read_log(run_dir)
        color = f"C{i % 10}"
        plt.plot(epochs, train, color=color, label=name)
        if any(v == v for v in val):
            plt.plot(epochs, val, "--", color=color)
    plt.xlabel("epoch")
    plt.ylabel("loss (solid = train, dashed = validation)")
    plt.yscale("log")
    plt.legend(fontsize=6)
    plt.grid(alpha=0.3)
    out = os.path.join(C.RUNS_DIR, "loss_curves.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def setting_of(row, key): # value of a setting for one results.csv row (the default from config.py if it was not overridden)
    overrides = json.loads(row.get("overrides") or "{}")
    return overrides.get(key, C._DEFAULTS[key])


def same_value(setting, wanted): # 1e-3 == 0.001 == "0.001"
    try:
        return float(setting) == float(wanted)
    except (TypeError, ValueError):
        return str(setting) == str(wanted)


def plot_metric(metric, vs, where): # one point per run: metric against a setting, e.g. error against TRAIN_FRACTION
    with open(os.path.join(C.RUNS_DIR, "results.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    points = []
    for row in rows:
        if any(not same_value(setting_of(row, k), v) for k, v in where.items()):
            continue
        if row.get(metric) in (None, ""):
            continue
        points.append((setting_of(row, vs), float(row[metric]), row["run_id"]))
    if not points:
        sys.exit(f"no runs with a value for '{metric}' (after --where), check the names with: head -1 {C.RUNS_DIR}/results.csv")
    numeric = all(isinstance(x, (int, float)) for x, _, _ in points)
    points.sort(key=lambda p: p[0] if numeric else str(p[0]))
    xs = [p[0] for p in points] if numeric else [str(p[0]) for p in points]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [p[1] for p in points], "o-")
    if numeric and len(set(xs)) > 2 and max(xs) / max(min(xs), 1e-12) > 20: # spans decades (learning rates)
        plt.xscale("log")
    plt.xlabel(vs)
    plt.ylabel(metric)
    note = ", ".join(f"{k}={v}" for k, v in where.items())
    plt.title(f"{metric} vs {vs}" + (f"  ({note})" if note else ""), fontsize=9)
    plt.grid(alpha=0.3)
    out = os.path.join(C.RUNS_DIR, f"{metric}_vs_{vs}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {out}  ({len(points)} runs)")
    if len({x for x in xs}) < len(xs):
        print("note: several runs share the same value, other settings differ between them; narrow with --where KEY=VALUE")


def main(argv):
    argv = C.apply_cli(argv) # --set RUNS_DIR=... works here too
    run_names, metric, vs, where, i = None, None, None, {}, 0
    while i < len(argv):
        a = argv[i]
        if a == "--runs":
            run_names, i = argv[i + 1].split(","), i + 2
        elif a == "--metric":
            metric, i = argv[i + 1], i + 2
        elif a == "--vs":
            vs, i = argv[i + 1].upper(), i + 2
        elif a == "--where":
            key, _, value = argv[i + 1].partition("=")
            where[key.upper()], i = value, i + 2
        else:
            sys.exit(f"unknown option '{a}', see the top of src/plot_runs.py")
    if metric or vs:
        if not (metric and vs):
            sys.exit("--metric and --vs go together")
        if vs not in C._DEFAULTS:
            sys.exit(f"unknown setting '{vs}'")
        plot_metric(metric, vs, where)
    else:
        if run_names is None:
            run_names = sorted(d for d in os.listdir(C.RUNS_DIR) if os.path.isdir(os.path.join(C.RUNS_DIR, d)))
        plot_all_losses(run_names)


if __name__ == "__main__":
    main(sys.argv[1:])
