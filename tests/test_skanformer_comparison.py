"""
Tests for the SKANFormer / transformer comparison.

    pip install pytest
    python -m pytest tests -q              # everything (~1 min on CPU)
    python -m pytest tests -q -m "not slow"  # skip the end-to-end training run

All tests run on CPU, no GPU needed.
"""

import copy
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.make_datasets import build_env, make  # noqa: E402
from experiments.analyze_results import analyze, parse_log  # noqa: E402
from src.model.sinekan import SineKANLayer  # noqa: E402
from src.model import skanformer, transformer  # noqa: E402
from src.dataset import EnvDataset  # noqa: E402


def small_params(operation="gcd", **overrides):
    env, params = build_env(operation, ["--cpu", "true", "--num_workers", "0",
                                        "--enc_emb_dim", "32", "--dec_emb_dim", "32",
                                        "--n_enc_layers", "2", "--n_dec_layers", "2",
                                        "--n_enc_heads", "4", "--n_dec_heads", "4"])
    for k, v in overrides.items():
        setattr(params, k, v)
    return env, params


def build(module, env, params):
    enc = module.TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
    dec = module.TransformerModel(params, env.id2word, is_encoder=False, with_output=True)
    return enc, dec


def n_params(*mods):
    return sum(p.numel() for m in mods for p in m.parameters() if p.requires_grad)


def batch(env, n=8, seed=0):
    rng = np.random.RandomState(seed)
    env.rng = rng
    xs, ys = zip(*[env.gen_expr() for _ in range(n)])

    def pad(seqs):
        lens = torch.LongTensor([len(s) + 2 for s in seqs])
        t = torch.full((int(lens.max()), len(seqs)), env.pad_index, dtype=torch.long)
        for i, s in enumerate(seqs):
            t[0, i] = env.eos_index
            t[1:len(s) + 1, i] = torch.LongTensor([env.word2id[w] for w in s])
            t[len(s) + 1, i] = env.eos_index
        return t, lens

    return pad(xs), pad(ys)


def run(enc, dec, x, xl, y, yl):
    enc.eval(), dec.eval()
    with torch.no_grad():
        e = enc("fwd", x=x, lengths=xl, causal=False).transpose(0, 1)
        return dec("fwd", x=y, lengths=yl, causal=True, src_enc=e, src_len=xl)


# ----------------------------------------------------------------------------
# SineKAN layer
# ----------------------------------------------------------------------------
def test_sinekan_layer_cpu_shapes_and_grad():
    layer = SineKANLayer(16, 24, device="cpu")
    x = torch.randn(5, 3, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (5, 3, 24)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.amplitudes.grad is not None and layer.freq.grad is not None
    # phase is a buffer: it must follow the module (to .cuda() in build_modules)
    assert "phase" in dict(layer.named_buffers())


# ----------------------------------------------------------------------------
# the baseline really is the original transformer
# ----------------------------------------------------------------------------
def test_sinekan_false_is_the_original_transformer():
    env, params = small_params(sinekan=False, last_layer_residual=True)
    torch.manual_seed(0)
    enc_s, dec_s = build(skanformer, env, params)
    enc_t, dec_t = build(transformer, env, params)
    # same parameters, same names, same shapes
    sd_s, sd_t = dec_s.state_dict(), dec_t.state_dict()
    assert sd_s.keys() == sd_t.keys()
    assert all(sd_s[k].shape == sd_t[k].shape for k in sd_s)
    # same function once the weights are shared
    enc_t.load_state_dict(enc_s.state_dict())
    dec_t.load_state_dict(dec_s.state_dict())
    (x, xl), (y, yl) = batch(env)
    assert torch.allclose(run(enc_s, dec_s, x, xl, y, yl), run(enc_t, dec_t, x, xl, y, yl), atol=1e-6)


def test_skanformer_replaces_only_last_decoder_ffn_with_similar_size():
    env, params = small_params(sinekan=True, last_layer_residual=True)
    enc, dec = build(skanformer, env, params)
    assert isinstance(dec.layers[-1].ffn, SineKANLayer)
    assert not any(isinstance(l.ffn, SineKANLayer) for l in dec.layers[:-1])
    assert not any(isinstance(l.ffn, SineKANLayer) for l in enc.layers)
    env, base = small_params(sinekan=False)
    enc_b, dec_b = build(skanformer, env, base)
    # SineKAN (dim*dim*grid) vs FFN (2*dim*4dim): comparable sizes -> a fair comparison
    assert abs(n_params(enc, dec) - n_params(enc_b, dec_b)) / n_params(enc_b, dec_b) < 0.02


def test_last_layer_residual_flag_changes_the_model():
    env, params = small_params(sinekan=True, last_layer_residual=True)
    torch.manual_seed(0)
    enc, dec = build(skanformer, env, params)
    params2 = copy.copy(params)
    params2.last_layer_residual = False
    enc2, dec2 = build(skanformer, env, params2)
    enc2.load_state_dict(enc.state_dict())
    dec2.load_state_dict(dec.state_dict())
    (x, xl), (y, yl) = batch(env)
    out1, out2 = run(enc, dec, x, xl, y, yl), run(enc2, dec2, x, xl, y, yl)
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all()
    assert not torch.allclose(out1, out2)


# ----------------------------------------------------------------------------
# data / evaluation plumbing used by the comparison
# ----------------------------------------------------------------------------
def test_data_mode_decodes_ranks_above_one():
    env, params = small_params("data")
    for r in range(5):
        assert env.code_class(["+", "1"], [str(r)]) == r
        assert env.check_prediction(["+", "1"], [str(r)], [str(r)])[0] == 2
    assert env.check_prediction(["+", "1"], ["2"], ["1"])[0] != 2


def test_eval_size_minus_one_uses_whole_file(tmp_path):
    env, params = small_params("gcd", global_rank=0, n_gpu_per_node=1)
    f = tmp_path / "e.txt"
    f.write_text("V2 + 4 + 6\t+ 2\n" * 37)
    ds = EnvDataset(env, "arithmetic", train=False, params=params, path=str(f), size=-1, type="valid")
    assert len(ds) == 37


def test_make_datasets_gcd_labels_are_correct(tmp_path):
    files = make("gcd", str(tmp_path), n_test=500, seed=3)
    env, _ = build_env("gcd")
    seen = set()
    for path in files:
        lines = open(path).read().splitlines()
        assert len(lines) == 500
        for line in lines:
            x, y = line.split("\t")
            a, b = env.input_encoder.decode(x.split())
            g = env.output_encoder.decode(y.split())
            assert math.gcd(int(a), int(b)) == g
            assert all(w in env.word2id for w in x.split() + y.split())
            if "balanced" in path:
                seen.add(g)
    assert max(seen) <= 100 and len(seen) > 90  # gcd spread over 1..100
    # deterministic
    files2 = make("gcd", str(tmp_path / "again"), n_test=500, seed=3)
    assert open(files[0]).read() == open(files2[0]).read()


def test_analyze_results_parses_logs(tmp_path):
    for model, accs in [("transformer", [10.0, 30.0]), ("skanformer", [20.0, 50.0])]:
        d = tmp_path / "gcd" / f"{model}-seed0"
        d.mkdir(parents=True)
        lines = [f"INFO - 01/01/26 10:00:00 - 0:00:01 - Number of parameters (encoder): 100",
                 f"INFO - 01/01/26 10:00:00 - 0:00:01 - Number of parameters (decoder): 50",
                 f"INFO - 01/01/26 10:00:00 - 0:00:05 -     200 -  900.00 examples/s -  9000.00 words/s"]
        for e, a in enumerate(accs):
            lines.append(f"INFO - 01/01/26 10:00:00 - 0:0{e + 1}:00 - __log__:"
                         f'{{"epoch": {e}, "valid_arithmetic_acc": {a}, "valid_arithmetic_xe_loss": 1.0, '
                         f'"valid_arithmetic_acc_1": {a}}}')
        (d / "train.log").write_text("\n".join(lines) + "\n")
    r = parse_log(str(tmp_path / "gcd" / "skanformer-seed0" / "train.log"))
    assert r["params"] == 150 and len(r["epochs"]) == 2 and r["speed"] == 900.0
    rows = analyze(str(tmp_path), quiet=True)
    by_model = {row["model"]: row for row in rows}
    assert by_model["skanformer"]["final_acc"] == 50.0
    assert by_model["transformer"]["best_acc"] == 30.0
    md = open(tmp_path / "results" / "summary.md", encoding="utf-8").read()
    assert "Δ skanformer − transformer" in md and "+20.00" in md


# ----------------------------------------------------------------------------
# end to end: the comparison script trains both models and reports
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_compare_models_smoke_end_to_end(tmp_path):
    tmp_path = tmp_path / "dir with spaces"  # e.g. "LLMs From Scratch" on Windows
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "compare_models.py"),
           "--preset", "smoke", "--cpu", "--max_epoch", "1", "--tasks", "gcd",
           "--models", "transformer,skanformer", "--dump_path", str(tmp_path)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    for model in ("transformer", "skanformer"):
        assert os.path.isfile(tmp_path / "gcd" / f"{model}-seed0" / "checkpoint.pth")
    summary = (tmp_path / "results" / "summary.csv").read_text()
    assert "transformer" in summary and "skanformer" in summary


# ----------------------------------------------------------------------------
# NaN guard in Trainer.optimize and the (modernised) Adam
# ----------------------------------------------------------------------------
def _fake_trainer(model, params):
    from types import SimpleNamespace
    from src.trainer import Trainer
    from src.optim import get_optimizer
    t = SimpleNamespace(params=params, optimizer=get_optimizer(list(model.parameters()), params.optimizer),
                        parameters={"model": list(model.parameters())}, n_iter=0)
    t._skip_nonfinite_step = lambda opt, what: Trainer._skip_nonfinite_step(t, opt, what)
    return t, (lambda loss: Trainer.optimize(t, loss))


def test_nan_loss_is_skipped_and_weights_stay_finite():
    _, params = small_params()
    model = torch.nn.Linear(4, 3)
    t, optimize = _fake_trainer(model, params)
    before = [p.detach().clone() for p in model.parameters()]
    optimize((model(torch.randn(2, 4)) * float("nan")).sum())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert t.n_nonfinite == 1
    # a normal step updates the weights (Adam with keyword-form ops) and resets the counter
    optimize(model(torch.randn(2, 4)).pow(2).sum())
    assert not all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters()) and t.n_nonfinite == 0


def test_repeated_nan_stops_training():
    _, params = small_params()
    model = torch.nn.Linear(4, 3)
    _, optimize = _fake_trainer(model, params)
    with pytest.raises(RuntimeError, match="diagnose_nan"):
        for _ in range(60):
            optimize(model(torch.full((1, 4), float("nan"))).sum())


def test_nan_report_pinpoints_the_layer(tmp_path, caplog):
    from types import SimpleNamespace
    from src.trainer import Trainer
    env, params = small_params(dump_path=str(tmp_path))
    enc, dec = build(skanformer, env, params)
    with torch.no_grad():
        dec.layers[1].ffn.lin1.weight[0, 0] = float("nan")   # break one weight
    (x, xl), (y, yl) = batch(env)
    t = SimpleNamespace(modules={"encoder": enc, "decoder": dec}, params=params,
                        n_total_iter=7, _last_batch=(x, xl, y, yl))
    with caplog.at_level("WARNING"):
        Trainer._nan_report(t, "loss")
    log = caplog.text
    assert "iteration 7" in log
    assert "decoder.layers.1.ffn.lin1.weight" in log                       # bad weight named
    assert "first non-finite layer output: decoder.layers.1.ffn" in log    # culprit layer
    assert os.path.isfile(tmp_path / "nan_batch.pt")
    assert "report failed" not in log
