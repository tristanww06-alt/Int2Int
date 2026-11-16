"""
Find where NaN / inf values come from when training on the GPU.

Runs one identical batch through the model on the GPU and on the CPU and reports:
  * PyTorch / CUDA / GPU versions,
  * whether the freshly initialised weights are finite,
  * the first layer whose output (forward) or gradient (backward) is NaN/inf,
  * the loss on GPU vs CPU, then a few real optimisation steps on the GPU,
  * the same GPU forward pass with TF32 and fused kernels disabled.

Usage (same model/task arguments as the failing run, all optional):
    python experiments/diagnose_nan.py
    python experiments/diagnose_nan.py --sinekan true
    python experiments/diagnose_nan.py --operation data --data_file data/elliptic_rank.train
Paste the whole output when asking for help.
"""

import argparse
import os
import platform
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_params(argv):
    from train import get_parser
    saved, sys.argv = sys.argv, [sys.argv[0], *argv]
    try:
        params = get_parser().parse_args(argv)
    finally:
        sys.argv = saved
    # fields normally set by init_distributed_mode()
    params.multi_gpu, params.global_rank, params.n_gpu_per_node = False, 0, 1
    return params


def make_batch(env, n, seed, data_file=None):
    rng = np.random.RandomState(seed)
    if data_file:
        with open(data_file, encoding="utf-8") as f:
            lines = [next(f).rstrip("\n") for _ in range(n)]
        pairs = [tuple(part.split() for part in l.split("\t")) for l in lines]
    else:
        env.rng = rng
        pairs = []
        while len(pairs) < n:
            xy = env.gen_expr()
            if xy is not None:
                pairs.append(xy)

    def pad(seqs):
        lens = torch.LongTensor([len(s) + 2 for s in seqs])
        t = torch.full((int(lens.max()), len(seqs)), env.pad_index, dtype=torch.long)
        for i, s in enumerate(seqs):
            t[0, i] = env.eos_index
            t[1:len(s) + 1, i] = torch.LongTensor([env.word2id[w] for w in s])
            t[len(s) + 1, i] = env.eos_index
        return t, lens

    (x1, len1), (x2, len2) = pad([p[0] for p in pairs]), pad([p[1] for p in pairs])
    return x1, len1, x2, len2


def loss_fn(encoder, decoder, x1, len1, x2, len2):
    alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
    pred_mask = alen[:, None] < len2[None] - 1
    y = x2[1:].masked_select(pred_mask[:-1])
    encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
    decoded = decoder("fwd", x=x2, lengths=len2, causal=True,
                      src_enc=encoded.transpose(0, 1), src_len=len1)
    _, loss = decoder("predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)
    return loss


def first_nonfinite(modules, run):
    """Run `run()` with hooks; return (forward culprit, backward culprit)."""
    found = {"fwd": None, "bwd": None}
    handles = []

    def bad(t):
        if isinstance(t, (tuple, list)):
            return any(bad(u) for u in t)
        return torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all()

    for mname, mod in modules.items():
        for name, sub in mod.named_modules():
            full = f"{mname}.{name}" if name else mname

            def fwd_hook(m, inp, out, full=full):
                if found["fwd"] is None and bad(out):
                    found["fwd"] = f"{full} ({type(m).__name__})" + \
                        ("  <- its INPUT was already non-finite" if bad(inp) else "")
                # gradient w.r.t. this module's output (tensor hook: works with in-place ops)
                if torch.is_tensor(out) and out.requires_grad:
                    def grad_hook(g, full=full, m=m):
                        if bad(g):
                            # backward runs from the loss down: keep the LAST one = closest to the loss
                            found["bwd"] = found["bwd"] or f"{full} ({type(m).__name__})"
                    out.register_hook(grad_hook)

            handles.append(sub.register_forward_hook(fwd_hook))
    try:
        loss = run()
    finally:
        for h in handles:
            h.remove()
    return loss, found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--data_file", default="", help="read the batch from a data file (operation data)")
    p.add_argument("--steps", type=int, default=30, help="optimisation steps to try on the GPU")
    args, train_args = p.parse_known_args()
    if not any(a == "--operation" for a in train_args):
        train_args = ["--operation", "gcd", *train_args]

    print("=" * 70)
    print(f"python {platform.python_version()} | torch {torch.__version__} | numpy {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()} | torch built for CUDA {torch.version.cuda}"
          f" | cuDNN {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} | compute capability {cap[0]}.{cap[1]}"
              f" | arch list of this torch build: {torch.cuda.get_arch_list()}")
    print(f"train.py args: {' '.join(train_args)}")
    print("=" * 70)

    import src.utils
    from src.envs import build_env
    from src.model import skanformer
    from src.optim import get_optimizer

    params = make_params([*train_args, "--cpu", "true"])  # build on CPU first
    env = build_env(params)
    torch.manual_seed(params.seed if params.seed >= 0 else 0)
    enc = skanformer.TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
    dec = skanformer.TransformerModel(params, env.id2word, is_encoder=False, with_output=True)
    n_par = sum(p.numel() for m in (enc, dec) for p in m.parameters())
    print(f"model built: {n_par:,} parameters, sinekan={params.sinekan}, "
          f"last_layer_residual={params.last_layer_residual}")
    bad_init = [n for m, mn in ((enc, "encoder"), (dec, "decoder"))
                for n, t in list(m.named_parameters()) + list(m.named_buffers())
                if t.is_floating_point() and not torch.isfinite(t).all()]
    print("initial weights finite:", "YES" if not bad_init else f"NO -> {bad_init}")

    batch = make_batch(env, args.batch, 0, args.data_file or None)
    results = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        src.utils.CUDA = dev == "cuda"
        e, d = copy_to(enc, dev), copy_to(dec, dev)
        e.train(), d.train()
        b = [t.to(dev) for t in batch]

        def run():
            l = loss_fn(e, d, *b)
            if torch.isfinite(l):
                l.backward()
            return l

        loss, found = first_nonfinite({"encoder": e, "decoder": d}, run)
        gnorm = torch.sqrt(sum((p.grad.float() ** 2).sum() for m in (e, d) for p in m.parameters()
                               if p.grad is not None)) if torch.isfinite(loss) else float("nan")
        results[dev] = loss.item()
        print(f"\n[{dev}] loss = {loss.item():.6f} | grad norm = {float(gnorm):.4f}")
        print(f"[{dev}] first non-finite forward output : {found['fwd'] or 'none'}")
        print(f"[{dev}] first non-finite gradient (from the loss side): {found['bwd'] or 'none'}")

    if "cuda" in results:
        diff = abs(results["cuda"] - results["cpu"])
        print(f"\nGPU vs CPU loss difference: {diff:.2e} ({'OK' if diff < 1e-2 else 'MISMATCH'})")

        # same forward with TF32 / fused attention disabled
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = False
        e, d = copy_to(enc, "cuda"), copy_to(dec, "cuda")
        with torch.no_grad():
            l = loss_fn(e, d, *[t.cuda() for t in batch])
        print(f"[cuda, TF32 + cuDNN off] loss = {l.item():.6f}")
        torch.backends.cudnn.enabled = True

        # a few real training steps on the GPU (same optimizer as train.py)
        e, d = copy_to(enc, "cuda"), copy_to(dec, "cuda")
        e.train(), d.train()
        opt = get_optimizer(list(e.parameters()) + list(d.parameters()), params.optimizer)
        print(f"\n{args.steps} training steps on the GPU ({params.optimizer}):")
        for step in range(args.steps):
            b = [t.cuda() for t in make_batch(env, args.batch, step + 1, None if not args.data_file else args.data_file)]
            opt.zero_grad()
            l = loss_fn(e, d, *b)
            if not torch.isfinite(l):
                _, found = first_nonfinite({"encoder": e, "decoder": d}, lambda: loss_fn(e, d, *b))
                w_bad = [n for m in (e, d) for n, t in m.named_parameters() if not torch.isfinite(t).all()]
                print(f"  step {step}: loss is {l.item()} | first bad layer: {found['fwd']}"
                      f" | non-finite weights: {w_bad[:5] or 'none'}")
                break
            l.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(e.parameters()) + list(d.parameters()), params.clip_grad_norm)
            if step % 5 == 0 or not torch.isfinite(gn):
                print(f"  step {step}: loss {l.item():.4f} grad norm {gn.item():.3f}")
            if not torch.isfinite(gn):
                print("  -> non-finite gradient")
                break
            opt.step()
        else:
            print("  all steps finite")
    print("\nDone. Please paste everything above.")


def copy_to(module, dev):
    import copy
    return copy.deepcopy(module).to(dev)


if __name__ == "__main__":
    main()
