"""
Collect the results of experiments/compare_models.py and write, in
<dump_path>/results/:

    summary.md       human readable comparison (also printed)
    summary.csv      one row per (task, eval set, model): final / best accuracy, loss, speed, size
    epochs.csv       every metric of every epoch of every run (for your own plots)
    per_class.csv    final accuracy per output class (elliptic rank, gcd value, ...)
    plots/*.png      accuracy and loss curves, per-class accuracy (needs matplotlib)

Usage:
    python experiments/analyze_results.py --dump_path runs
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

LOG_RE = re.compile(r"__log__:(\{.*\})\s*$")
ELAPSED_RE = re.compile(r" - (?:(\d+) days?, )?(\d+):(\d\d):(\d\d) - ")
SPEED_RE = re.compile(r"([\d.]+) examples/s")
PARAMS_RE = re.compile(r"Number of parameters \((\w+)\): (\d+)")
EXP_RE = re.compile(r"^(?P<model>.+)-seed(?P<seed>\d+)$")

# fixed color per model (identity never depends on which models are present)
MODEL_COLORS = {
    "transformer": "#2a78d6",
    "skanformer": "#eb6834",
    "skanformer_nores": "#1baf7a",
}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_ORDER = ["transformer", "skanformer", "skanformer_nores"]

# names of the eval sets written by compare_models.py
EVAL_SET_NAMES = {
    ("gcd", "valid"): "natural distribution",
    ("gcd", "test"): "balanced gcd 1..100",
    ("elliptic_rank", "valid"): "LMFDB test set",
}


def color_for(model, models):
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    others = [m for m in sorted(models) if m not in MODEL_COLORS]
    return EXTRA_COLORS[others.index(model) % len(EXTRA_COLORS)]


def parse_log(path):
    """Parse one train.log. Returns dict(epochs=[scores...], params, speed, minutes)."""
    epochs, speeds, n_params = [], [], {}
    total_s, seg_last = 0.0, 0.0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ELAPSED_RE.search(line)
            if m:
                d, h, mi, s = m.groups()
                t = int(d or 0) * 86400 + int(h) * 3600 + int(mi) * 60 + int(s)
                if t < seg_last:  # log appended by a resumed run: elapsed restarts at 0
                    total_s += seg_last
                seg_last = t
            m = LOG_RE.search(line)
            if m:
                sc = json.loads(m.group(1))
                sc["_minutes"] = (total_s + seg_last) / 60
                epochs.append(sc)
                continue
            m = SPEED_RE.search(line)
            if m:
                speeds.append(float(m.group(1)))
            m = PARAMS_RE.search(line)
            if m:
                n_params[m.group(1)] = int(m.group(2))
    # de-duplicate epochs (a resumed run may re-log an epoch), keep the last occurrence
    by_epoch = {}
    for sc in epochs:
        by_epoch[sc.get("epoch", len(by_epoch))] = sc
    epochs = [by_epoch[k] for k in sorted(by_epoch)]
    return dict(
        epochs=epochs,
        params=sum(n_params.values()) if n_params else None,
        params_detail=n_params,
        speed=(sum(speeds[1:]) / len(speeds[1:])) if len(speeds) > 1 else (speeds[0] if speeds else None),
        minutes=(total_s + seg_last) / 60,
    )


def collect(dump_path):
    """All runs: list of dict(task, model, seed, path, **parse_log)."""
    runs = []
    for task in sorted(os.listdir(dump_path)):
        tdir = os.path.join(dump_path, task)
        if task in ("results", "data") or not os.path.isdir(tdir):
            continue
        for exp_id in sorted(os.listdir(tdir)):
            log = os.path.join(tdir, exp_id, "train.log")
            if not os.path.isfile(log):
                continue
            m = EXP_RE.match(exp_id)
            model, seed = (m.group("model"), int(m.group("seed"))) if m else (exp_id, 0)
            r = parse_log(log)
            if not r["epochs"]:
                continue
            runs.append(dict(task=task, model=model, seed=seed, path=log, **r))
    return runs


def eval_sets(scores):
    """'valid', 'test', 'test2', ... present in an epoch's scores."""
    sets = sorted({k.split("_arithmetic_")[0] for k in scores if "_arithmetic_acc" in k})
    return sorted(sets, key=lambda s: (s != "valid", s))


def class_keys(scores, es):
    """per-class accuracy keys, e.g. valid_arithmetic_acc_3 -> '3' (sorted numerically)."""
    pre = f"{es}_arithmetic_acc_"
    out = []
    for k in scores:
        if k.startswith(pre):
            c = k[len(pre):]
            if re.fullmatch(r"\d+(-\d+)?", c):
                out.append(c)
    return sorted(out, key=lambda c: [int(x) for x in c.split("-")])


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


def fmt(m, s=None, digits=2):
    if m is None:
        return "-"
    if s is None or s == 0:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def ordered_models(models):
    return [m for m in MODEL_ORDER if m in models] + sorted(m for m in models if m not in MODEL_ORDER)


def summarize(runs):
    """rows of summary.csv"""
    rows = []
    groups = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["model"])].append(r)
    for (task, model), rs in sorted(groups.items()):
        for es in eval_sets(rs[0]["epochs"][-1]):
            acc_key = f"{es}_arithmetic_acc"
            final = [r["epochs"][-1].get(acc_key) for r in rs]
            best, best_ep = [], []
            for r in rs:
                accs = [e.get(acc_key, float("nan")) for e in r["epochs"]]
                i = max(range(len(accs)), key=lambda j: accs[j])
                best.append(accs[i])
                best_ep.append(r["epochs"][i].get("epoch", i))
            loss = [r["epochs"][-1].get(f"{es}_arithmetic_xe_loss") for r in rs]
            fm, fs = mean_std(final)
            bm, bs = mean_std(best)
            lm, ls = mean_std(loss)
            row = dict(
                task=task, eval_set=es, eval_name=EVAL_SET_NAMES.get((task, es), es), model=model,
                n_seeds=len(rs), epochs=min(len(r["epochs"]) for r in rs),
                params=rs[0]["params"],
                final_acc=fm, final_acc_std=fs, best_acc=bm, best_acc_std=bs,
                best_epoch=mean_std(best_ep)[0], final_xe_loss=lm, final_xe_loss_std=ls,
                examples_per_s=mean_std([r["speed"] for r in rs])[0],
                minutes_per_run=mean_std([r["minutes"] for r in rs])[0],
            )
            # gcd on the balanced set: number of gcd values 1..100 predicted with >= 90% accuracy
            if task == "gcd" and es == "test":
                n90 = [sum(1 for c in class_keys(r["epochs"][-1], es)
                           if r["epochs"][-1][f"{es}_arithmetic_acc_{c}"] >= 90 and int(c) <= 100)
                       for r in rs]
                row["gcd_values_ge90"] = mean_std(n90)[0]
            rows.append(row)
    return rows


def per_class_rows(runs):
    rows = []
    for r in runs:
        last = r["epochs"][-1]
        for es in eval_sets(last):
            for c in class_keys(last, es):
                rows.append(dict(task=r["task"], eval_set=es, model=r["model"], seed=r["seed"],
                                 epoch=last.get("epoch"), cls=c, acc=last[f"{es}_arithmetic_acc_{c}"]))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def markdown(runs, rows, pc_rows):
    out = ["# SKANFormer vs Transformer", ""]
    out.append("Accuracies are in %, mean ± std over seeds. *final* = last epoch, *best* = best epoch "
               "of each run. Loss is the cross-entropy on the eval set at the last epoch.")
    out.append("")
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for task, trs in by_task.items():
        out.append(f"## {task}")
        out.append("")
        has90 = any("gcd_values_ge90" in r for r in trs)
        hdr = ["eval set", "model", "seeds", "epochs", "params", "final acc", "best acc (epoch)",
               "final loss", "examples/s", "min/run"] + (["gcd ≤100 at ≥90%"] if has90 else [])
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for es in sorted({r["eval_set"] for r in trs}, key=lambda s: (s != "valid", s)):
            ers = {r["model"]: r for r in trs if r["eval_set"] == es}
            for model in ordered_models(ers):
                r = ers[model]
                cells = [r["eval_name"], model, str(r["n_seeds"]), str(r["epochs"]),
                         f"{r['params']:,}" if r["params"] else "-",
                         fmt(r["final_acc"], r["final_acc_std"]),
                         f"{fmt(r['best_acc'], r['best_acc_std'])} ({r['best_epoch']:.0f})",
                         fmt(r["final_xe_loss"], r["final_xe_loss_std"], 4),
                         fmt(r["examples_per_s"], None, 0), fmt(r["minutes_per_run"], None, 1)]
                if has90:
                    cells.append(fmt(r.get("gcd_values_ge90"), None, 1))
                out.append("| " + " | ".join(cells) + " |")
            if "transformer" in ers:
                base = ers["transformer"]
                for model in ordered_models(ers):
                    if model == "transformer":
                        continue
                    r = ers[model]
                    if r["final_acc"] is None or base["final_acc"] is None:
                        continue
                    out.append(f"| {r['eval_name']} | **Δ {model} − transformer** | | | | "
                               f"**{r['final_acc'] - base['final_acc']:+.2f}** | "
                               f"**{r['best_acc'] - base['best_acc']:+.2f}** | "
                               f"{r['final_xe_loss'] - base['final_xe_loss']:+.4f} | | |"
                               + (" |" if has90 else ""))
        out.append("")

        # per class table (mean over seeds), restricted to a readable number of classes
        pcs = [p for p in pc_rows if p["task"] == task]
        if pcs:
            for es in sorted({p["eval_set"] for p in pcs}, key=lambda s: (s != "valid", s)):
                ps = [p for p in pcs if p["eval_set"] == es]
                classes = sorted({p["cls"] for p in ps}, key=lambda c: [int(x) for x in c.split("-")])
                if task == "elliptic_rank":
                    label = "rank"
                else:
                    label = "gcd" if task == "gcd" else "class"
                    classes = classes[:12]
                models = ordered_models({p["model"] for p in ps})
                agg = defaultdict(list)
                for p in ps:
                    agg[(p["model"], p["cls"])].append(p["acc"])
                out.append(f"Final accuracy per {label} ({EVAL_SET_NAMES.get((task, es), es)}"
                           f"{', first 12 classes' if task != 'elliptic_rank' else ''}):")
                out.append("")
                out.append(f"| model | " + " | ".join(f"{label} {c}" for c in classes) + " |")
                out.append("|" + "|".join(["---"] * (len(classes) + 1)) + "|")
                for m in models:
                    out.append(f"| {m} | " + " | ".join(fmt(mean_std(agg[(m, c)])[0], None, 1)
                                                         if agg[(m, c)] else "-" for c in classes) + " |")
                out.append("")
    out.append(f"{len(runs)} runs; logs in <dump_path>/<task>/<model>-seed<k>/train.log")
    out.append("")
    return "\n".join(out)


def plot(runs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not installed: skipping plots (pip install matplotlib)")
        return []
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    })
    files = []
    tasks = sorted({r["task"] for r in runs})
    all_models = {r["model"] for r in runs}
    for task in tasks:
        trs = [r for r in runs if r["task"] == task]
        sets = eval_sets(trs[0]["epochs"][-1])
        models = ordered_models({r["model"] for r in trs})
        for metric, ylabel, fname in [("acc", "accuracy (%)", "accuracy"),
                                      ("xe_loss", "eval cross-entropy", "loss")]:
            fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 3.6), squeeze=False)
            for ax, es in zip(axes[0], sets):
                key = f"{es}_arithmetic_{metric}"
                for model in models:
                    rs = [r for r in trs if r["model"] == model]
                    n = min(len(r["epochs"]) for r in rs)
                    xs = [rs[0]["epochs"][i].get("epoch", i) + 1 for i in range(n)]
                    vals = [[r["epochs"][i].get(key, float("nan")) for r in rs] for i in range(n)]
                    mean = [sum(v) / len(v) for v in vals]
                    c = color_for(model, all_models)
                    ax.plot(xs, mean, color=c, lw=2, marker="o", ms=4,
                            label=f"{model} ({len(rs)} seed{'s' if len(rs) > 1 else ''})")
                    if len(rs) > 1:
                        ax.fill_between(xs, [min(v) for v in vals], [max(v) for v in vals],
                                        color=c, alpha=0.15, lw=0)
                ax.set_title(f"{task}: {EVAL_SET_NAMES.get((task, es), es)}", fontsize=10, loc="left")
                ax.set_xlabel("epoch")
                ax.set_ylabel(ylabel)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if metric == "acc":
                    ax.set_ylim(bottom=0)
                ax.legend(fontsize=9)
            fig.tight_layout()
            f = os.path.join(out_dir, f"{task}_{fname}.png")
            fig.savefig(f, dpi=150)
            plt.close(fig)
            files.append(f)

        # per class final accuracy (grouped bars), first eval set
        es = sets[0]
        classes = class_keys(trs[0]["epochs"][-1], es)
        if task != "elliptic_rank":
            classes = classes[:15]
        if classes:
            fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(classes) * len(models)), 3.6))
            w = 0.8 / len(models)
            for j, model in enumerate(models):
                rs = [r for r in trs if r["model"] == model]
                ys = [sum(r["epochs"][-1].get(f"{es}_arithmetic_acc_{c}", 0) for r in rs) / len(rs)
                      for c in classes]
                ax.bar([i + (j - (len(models) - 1) / 2) * w for i in range(len(classes))], ys,
                       width=w * 0.92, color=color_for(model, all_models), label=model)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_xlabel({"elliptic_rank": "rank", "gcd": "true gcd"}.get(task, "class (true output)"))
            ax.set_ylabel("final accuracy (%)")
            ax.set_ylim(0, 100)
            ax.set_title(f"{task}: accuracy per class, last epoch ({EVAL_SET_NAMES.get((task, es), es)})",
                         fontsize=10, loc="left")
            ax.legend(fontsize=9)
            fig.tight_layout()
            f = os.path.join(out_dir, f"{task}_per_class.png")
            fig.savefig(f, dpi=150)
            plt.close(fig)
            files.append(f)
    return files


def analyze(dump_path, quiet=False):
    dump_path = os.path.abspath(os.path.expanduser(dump_path))
    runs = collect(dump_path)
    if not runs:
        print(f"No finished epoch found under {dump_path}")
        return None
    res = os.path.join(dump_path, "results")
    os.makedirs(res, exist_ok=True)

    epoch_rows = []
    for r in runs:
        for sc in r["epochs"]:
            row = dict(task=r["task"], model=r["model"], seed=r["seed"], params=r["params"])
            row.update({k: v for k, v in sc.items()})
            epoch_rows.append(row)
    write_csv(os.path.join(res, "epochs.csv"), epoch_rows)
    rows = summarize(runs)
    write_csv(os.path.join(res, "summary.csv"), rows)
    pc = per_class_rows(runs)
    write_csv(os.path.join(res, "per_class.csv"), pc)
    md = markdown(runs, rows, pc)
    plots = plot(runs, os.path.join(res, "plots"))
    if plots:
        md += "\n## Plots\n\n" + "\n".join(f"![{os.path.basename(p)}](plots/{os.path.basename(p)})"
                                            for p in plots) + "\n"
    with open(os.path.join(res, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    if not quiet:
        print("\n" + md)
        print(f"Results written in {res}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_path", required=True)
    analyze(p.parse_args().dump_path)


if __name__ == "__main__":
    main()
