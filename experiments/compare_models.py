"""
Compare the SKANFormer (SineKAN as the last decoder FFN) to the plain
transformer on Int2Int problems: GCD, elliptic curve rank, and optionally the
other built-in arithmetic tasks.

Each (task, model, seed) is one ordinary ``train.py`` run, written to
``<dump_path>/<task>/<model>-seed<k>/``. Both models of a given seed get the
same model-init seed, the same training-data stream (``--env_base_seed``) and
the same fixed evaluation files, so the only difference is the architecture.
When all runs are done, ``experiments/analyze_results.py`` is called to write
tables and plots in ``<dump_path>/results/``.

Re-running the same command resumes: finished runs are skipped, interrupted
runs continue from their last checkpoint.

Quick start (from the repository root):

    # 0. elliptic curve data lives in git LFS
    git lfs pull

    # 1. 2-minute sanity check that everything runs (works on CPU too: add --cpu)
    python experiments/compare_models.py --preset smoke --dump_path runs_smoke

    # 2. the actual comparison (default preset: 300k examples/epoch, 10 epochs)
    python experiments/compare_models.py --dump_path runs

    # more seeds / more models / more tasks
    python experiments/compare_models.py --dump_path runs --seeds 0,1,2 \\
        --models transformer,skanformer,skanformer_nores \\
        --tasks gcd,elliptic_rank,modular_mul

Any argument the script does not know is forwarded to train.py, e.g.
``--batch_size 128`` or ``--optimizer adam,lr=0.0003``.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------
# models: architecture flags passed to train.py (src/model/skanformer.py)
# --------------------------------------------------------------------------
MODELS = {
    # with --sinekan false the skanformer code is exactly the original transformer
    "transformer": ["--sinekan", "false"],
    # last decoder FFN replaced by a SineKAN layer, residual connection kept
    "skanformer": ["--sinekan", "true", "--last_layer_residual", "true"],
    # same, without the residual connection around the SineKAN layer
    "skanformer_nores": ["--sinekan", "true", "--last_layer_residual", "false"],
}

# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
# "generated" tasks are trained on data generated on the fly (as in the paper)
# and evaluated on fixed files written by make_datasets.py.
GENERATED_TASKS = {
    "gcd": [],
    "modular_add": [],
    "modular_mul": [],
    "fraction_compare": [],
    "fraction_simplify": [],
    "fraction_add": [],
    "matrix_rank": ["--dim1", "5", "--dim2", "5", "--maxint", "10"],
}
FILE_TASKS = {"elliptic_rank"}
ALL_TASKS = list(GENERATED_TASKS) + sorted(FILE_TASKS)

# --------------------------------------------------------------------------
# presets (anything here can be overridden on the command line)
# --------------------------------------------------------------------------
PRESETS = {
    # does everything run? tiny model, a couple of thousand examples
    "smoke": dict(epoch_size=2000, max_epoch=2, n_eval=300, reload_size=20000,
                  model=["--enc_emb_dim", "64", "--dec_emb_dim", "64",
                         "--n_enc_layers", "1", "--n_dec_layers", "1"]),
    # first look, ~1/6 of the default budget
    "quick": dict(epoch_size=100000, max_epoch=5, n_eval=5000, reload_size=-1, model=[]),
    # default: paper epoch size and model (4+4 layers, dim 256, 8 heads), 10 epochs
    "standard": dict(epoch_size=300000, max_epoch=10, n_eval=10000, reload_size=-1, model=[]),
}


def is_lfs_pointer(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(64).startswith("version https://git-lfs")
    except OSError:
        return False


def completed_epochs(log_path):
    """Number of finished (evaluated) epochs recorded in a train.log."""
    if not os.path.isfile(log_path):
        return 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for line in f if "__log__:" in line)


def prepare_task(task, args, preset, extra):
    """Return (train.py args for this task, error message or None)."""
    if task in GENERATED_TASKS:
        from experiments.make_datasets import make

        task_args = GENERATED_TASKS[task]
        data_dir = os.path.join(args.dump_path, "data")
        valid = os.path.join(data_dir, f"{task}.valid.txt")
        n_eval = args.n_eval if args.n_eval is not None else preset["n_eval"]
        expected = [valid] + ([os.path.join(data_dir, "gcd_balanced.test.txt")] if task == "gcd" else [])
        # regenerate when the generation settings change; problem parameters
        # (e.g. --maxint, --base) are forwarded so the files match the training run
        meta_path = os.path.join(data_dir, f"{task}.meta.json")
        meta = {"n_eval": n_eval, "seed": args.data_seed, "args": task_args + extra}
        old_meta = None
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                old_meta = json.load(f)
        if old_meta != meta or any(not os.path.isfile(f) for f in expected):
            print(f"[data] writing fixed eval set(s) for {task} ({n_eval} examples) in {data_dir}")
            expected = make(task, data_dir, n_test=n_eval, seed=args.data_seed,
                            extra_args=task_args + extra)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        return ["--operation", task, *task_args,
                "--eval_data", ",".join(expected), "--eval_size", "-1"], None

    if task == "elliptic_rank":
        train = os.path.join(args.elliptic_dir, "elliptic_rank.train")
        test = os.path.join(args.elliptic_dir, "elliptic_rank.test")
        for f in (train, test):
            if not os.path.isfile(f):
                return None, f"{f} not found"
            if is_lfs_pointer(f):
                return None, (f"{f} is a git-lfs pointer, not the data. Run `git lfs install && "
                              f"git lfs pull` in the repository (or point --elliptic_dir to the files).")
        reload_size = args.reload_size if args.reload_size is not None else preset["reload_size"]
        n_eval = args.n_eval if args.n_eval is not None else (preset["n_eval"] if args.preset == "smoke" else -1)
        return ["--operation", "data", "--train_data", train, "--eval_data", test,
                "--eval_size", str(n_eval), "--reload_size", str(reload_size)], None

    return None, f"unknown task {task}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump_path", required=True, help="where runs, data and results are written")
    p.add_argument("--tasks", default="gcd,elliptic_rank", help=f"comma separated, from {ALL_TASKS}")
    p.add_argument("--models", default="transformer,skanformer", help=f"comma separated, from {list(MODELS)}")
    p.add_argument("--seeds", default="0", help="comma separated seeds, e.g. 0,1,2")
    p.add_argument("--preset", default="standard", choices=list(PRESETS))
    p.add_argument("--max_epoch", type=int, default=None, help="epochs per run (overrides preset)")
    p.add_argument("--epoch_size", type=int, default=None, help="training examples per epoch (overrides preset)")
    p.add_argument("--n_eval", type=int, default=None,
                   help="size of the fixed eval sets for generated tasks (elliptic: whole 10k test file)")
    p.add_argument("--reload_size", type=int, default=None,
                   help="elliptic: only load the first N training curves (-1 = all 1M)")
    p.add_argument("--elliptic_dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--data_seed", type=int, default=1234, help="seed for the fixed eval sets")
    p.add_argument("--cpu", action="store_true", help="train on CPU (slow, use with --preset smoke)")
    p.add_argument("--python", default=sys.executable, help="python interpreter used to run train.py")
    p.add_argument("--dry_run", action="store_true", help="print the train.py commands only")
    p.add_argument("--stop_on_error", action="store_true", help="abort at the first failed run")
    p.add_argument("--no_analysis", action="store_true", help="do not run analyze_results.py at the end")
    args, extra = p.parse_known_args()

    args.dump_path = os.path.abspath(os.path.expanduser(args.dump_path))
    preset = PRESETS[args.preset]
    max_epoch = args.max_epoch if args.max_epoch is not None else preset["max_epoch"]
    epoch_size = args.epoch_size if args.epoch_size is not None else preset["epoch_size"]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for m in models:
        if m not in MODELS:
            p.error(f"unknown model {m}, choose from {list(MODELS)}")
    for t in tasks:
        if t not in ALL_TASKS:
            p.error(f"unknown task {t}, choose from {ALL_TASKS}")
    os.makedirs(args.dump_path, exist_ok=True)

    common = ["--epoch_size", str(epoch_size),
              "--validation_metrics", "valid_arithmetic_acc",  # keeps best-valid_arithmetic_acc.pth
              *preset["model"]]
    if args.cpu:
        common += ["--cpu", "true", "--num_workers", "0"]

    status = []
    for task in tasks:
        task_args, err = prepare_task(task, args, preset, extra)
        if err:
            print(f"[skip] {task}: {err}")
            status.append((task, "*", "*", "SKIPPED: " + err))
            if args.stop_on_error:
                sys.exit(1)
            continue
        for seed in seeds:
            for model in models:
                exp_id = f"{model}-seed{seed}"
                log_path = os.path.join(args.dump_path, task, exp_id, "train.log")
                done = completed_epochs(log_path)
                if done >= max_epoch:
                    print(f"[done] {task}/{exp_id} ({done} epochs)")
                    status.append((task, model, seed, f"done ({done} epochs)"))
                    continue
                cmd = [args.python, "train.py",
                       "--dump_path", args.dump_path, "--exp_name", task, "--exp_id", exp_id,
                       "--seed", str(seed), "--env_base_seed", str(seed),
                       "--max_epoch", str(max_epoch - done),  # train.py resumes from checkpoint.pth
                       *common, *task_args, *MODELS[model], *extra]
                print(f"\n[run] {task}/{exp_id}" + (f" (resuming after epoch {done})" if done else ""))
                print("      " + " ".join(shlex.quote(c) for c in cmd))
                if args.dry_run:
                    continue
                t0 = time.time()
                ret = subprocess.call(cmd, cwd=ROOT)
                dt = (time.time() - t0) / 60
                if ret != 0:
                    msg = f"FAILED (exit code {ret}, see {log_path})"
                    print(f"[error] {task}/{exp_id}: {msg}")
                    status.append((task, model, seed, msg))
                    if args.stop_on_error:
                        sys.exit(ret)
                else:
                    status.append((task, model, seed, f"trained in {dt:.1f} min"))

    print("\n=========== summary of runs ===========")
    for t, m, s, msg in status:
        print(f"{t:18s} {str(m):18s} seed {s}: {msg}")

    if not args.dry_run and not args.no_analysis:
        from experiments.analyze_results import analyze
        analyze(args.dump_path)


if __name__ == "__main__":
    main()
