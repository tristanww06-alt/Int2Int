"""
Generate *fixed* evaluation (and optionally training) files for the generated
Int2Int problems (gcd, modular_add, modular_mul, fraction_compare, ...).

Why: when Int2Int generates data on the fly, the validation set is re-drawn at
random every epoch (``RandomState(None)`` in src/dataset.py). To compare two
architectures fairly, both should be scored on exactly the same examples, so we
write them to disk once and pass them with ``--eval_data``.

The examples are produced with the repository's own generator and encoders
(src/envs/arithmetic.py + generators.py), so they follow exactly the training
distribution and tokenisation.

For gcd we additionally write a "balanced" test set, in which the GCD is
uniformly distributed over 1..100 (in the natural distribution ~61% of pairs
are coprime, so a model that always answers 1 already scores ~61%).

Examples:
    python experiments/make_datasets.py --operation gcd --out_dir runs/data
    python experiments/make_datasets.py --operation modular_mul --n_test 5000 --out_dir runs/data
"""

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_env(operation, extra_args=()):
    """Build an ArithmeticEnvironment with the default train.py parameters."""
    from train import get_parser
    from src.envs import build_env as _build_env

    argv = ["--operation", operation, *extra_args]
    # get_parser() peeks at sys.argv (parse_known_args) to pick the environment
    saved, sys.argv = sys.argv, [sys.argv[0], *argv]
    try:
        parser = get_parser()
        params = parser.parse_args(argv)
    finally:
        sys.argv = saved
    return _build_env(params), params


def generate_examples(env, n, seed):
    """n (input tokens, output tokens) pairs from the env generator."""
    env.rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        xy = env.gen_expr()
        if xy is None:
            continue
        out.append(xy)
    return out


def generate_balanced_gcd(env, params, n, seed, max_gcd=100):
    """GCD uniformly distributed in 1..max_gcd, inputs <= params.maxint."""
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        g = int(rng.randint(1, max_gcd + 1))
        hi = params.maxint // g
        if hi < 1:
            continue
        u, v = (int(t) for t in rng.randint(1, hi + 1, 2))
        if math.gcd(u, v) != 1:
            continue
        x = env.input_encoder.encode([u * g, v * g])
        y = env.output_encoder.encode(g)
        out.append((x, y))
    return out


def write(path, examples):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x, y in examples:
            f.write(" ".join(x) + "\t" + " ".join(y) + "\n")


def make(operation, out_dir, n_test=10000, n_train=0, seed=1234, extra_args=()):
    """
    Writes <out_dir>/<operation>.valid.txt (same distribution as training),
    <out_dir>/gcd_balanced.test.txt for gcd and, if n_train > 0,
    <out_dir>/<operation>.train.txt. Returns the list of eval files written.
    """
    env, params = build_env(operation, extra_args)
    files = []

    valid = os.path.join(out_dir, f"{operation}.valid.txt")
    write(valid, generate_examples(env, n_test, seed))
    files.append(valid)

    if operation == "gcd":
        bal = os.path.join(out_dir, "gcd_balanced.test.txt")
        write(bal, generate_balanced_gcd(env, params, n_test, seed + 1))
        files.append(bal)

    if n_train > 0:
        # different seed from the eval set; for gcd on 1..1e6 the overlap is negligible
        train = os.path.join(out_dir, f"{operation}.train.txt")
        write(train, generate_examples(env, n_train, seed + 2))

    return files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--operation", default="gcd")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--n_train", type=int, default=0, help="also write a fixed training file (0 = no)")
    p.add_argument("--seed", type=int, default=1234)
    args, extra = p.parse_known_args()
    files = make(args.operation, args.out_dir, args.n_test, args.n_train, args.seed, extra)
    for f in files:
        print("wrote", f)


if __name__ == "__main__":
    main()
