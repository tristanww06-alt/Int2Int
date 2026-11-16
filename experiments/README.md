# SKANFormer vs Transformer

Scripts to compare the SKANFormer (`--sinekan true`: the FFN of the last decoder
layer is a SineKAN layer, `src/model/sinekan.py`) to the original Int2Int
transformer (`--sinekan false`, identical to `src/model/transformer.py`).

| file | what it does |
|---|---|
| `compare_models.py` | runs `train.py` for every (task, model, seed), resumable, then calls the analysis |
| `make_datasets.py` | writes fixed eval sets for the generated tasks (gcd, modular_*, fraction_*, matrix_rank) |
| `analyze_results.py` | parses the `train.log` files into `results/summary.md`, CSVs and plots |
| `../tests/test_skanformer_comparison.py` | unit tests and an end-to-end smoke run (CPU, about 15 s) |

## Running it

```bash
pip install torch numpy matplotlib pytest
git lfs install && git lfs pull          # elliptic_rank.train / .test are stored in git LFS
python -m pytest -q                      # sanity checks, CPU only
python experiments/compare_models.py --preset smoke --dump_path runs_smoke   # about 2 min
python experiments/compare_models.py --dump_path runs                         # the comparison
```

Results go to `runs/results/summary.md`, along with `summary.csv`, `epochs.csv`,
`per_class.csv` and `plots/`. To recompute them at any time, run
`python experiments/analyze_results.py --dump_path runs`.

### What is compared

* **Models** (`--models`): `transformer`, `skanformer`, and optionally
  `skanformer_nores` (no residual connection around the SineKAN layer).
  With the default size (4+4 layers, dimension 256), the SineKAN layer has
  256·256·8 weights, which matches the 2·256·1024 of the FFN it replaces. The
  two models have nearly the same number of parameters.
* **Tasks** (`--tasks`), default `gcd,elliptic_rank`:
  * `gcd`: pairs of integers in 1..10^6 written in base 1000, generated on the
    fly as in the paper. Each epoch is evaluated on two fixed sets of 10k pairs.
    The *natural* set has about 61% coprime pairs, so always answering 1 already
    scores about 61%. In the *balanced* set the GCD is uniform over 1..100.
    The summary also counts how many GCD values in 1..100 are predicted with
    at least 90% accuracy.
  * `elliptic_rank`: the LMFDB dataset of the paper (1M training curves, 10k
    test curves), with accuracy reported per rank.
  * Also available: `modular_add`, `modular_mul`, `fraction_compare`,
    `fraction_simplify`, `fraction_add`, `matrix_rank`.
* **Fairness:** for a given seed, both models share the same initialisation seed,
  the same training-data stream (`--env_base_seed`) and the same eval files.
  Use `--seeds 0,1,2` to get error bars.

### Budget (presets)

| preset | examples/epoch | epochs | approx. time per run* |
|---|---|---|---|
| `smoke` | 2,000 (tiny model) | 2 | seconds |
| `quick` | 100,000 | 5 | ~10 min |
| `standard` (default) | 300,000 (as in the paper) | 10 | ~1 h |

\*At the 800–950 examples/s the paper reports for batch size 32. The default
comparison (2 tasks × 2 models × 1 seed) therefore takes about 4–5 h. For
reference, the paper reaches about 85% on GCD and 48% on elliptic rank after one
epoch. Any of these can be overridden, e.g. `--max_epoch 20 --epoch_size 200000`.
Any other argument is passed to `train.py` unchanged, e.g. `--batch_size 128`
(faster on a big GPU, but the result is no longer the paper's setting).

Re-running the same command resumes the runs: finished runs are skipped,
interrupted ones continue from `checkpoint.pth`, and raising `--max_epoch`
extends finished runs. Each run also keeps its best model in
`best-valid_arithmetic_acc.pth`.

## Fixes made to the codebase for this comparison

* `SineKANLayer` was built with `device='cuda'`, so the model could not be
  built on CPU. It is now built on CPU and moved to the GPU with the rest of the model.
* `--operation data` could only decode the outputs `0` and `1`, so any elliptic
  curve of rank 2 or more crashed evaluation. It now decodes 0..`--max_class`.
* `--eval_size -1` (the whole eval file) failed an assertion.
* A single `--eval_data` file made the evaluator ask for a second file that does
  not exist. Eval file 0 is now `valid` and files 1, 2, … are `test`, `test2`, ….
* New `--seed` flag in `train.py` that seeds the model initialisation.
