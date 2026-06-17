"""
PyTorch blocked_autorange benchmark for NLL loss.

Defaults to the MPS device (this kernel's target) but works on any device via
--device, so the same script can profile the Metal path against CUDA/CPU.
Compares Metal vs MPSGraph baselines (requires restoring old dylib for baseline).
Run after deploying new dylib.
"""

import argparse
import itertools

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]
LAYOUTS = ["contig", "noncontig"]
# (N=rows, C=num_classes) tied to real model output shapes rather than
# synthetic powers of two:
#   classification heads (C = #labels) and LM heads (C = vocab size).
CONFIGS = [
    (256, 1000),  # ImageNet-1k head, batch 256
    (1024, 21841),  # ImageNet-22k head
    (4096, 32000),  # Llama-2 vocab, a few sequences worth of tokens
    (2048, 50257),  # GPT-2 vocab
    (1024, 128256),  # Llama-3 vocab
]


def make_log_prob(N, C, dtype, device, layout):
    if layout == "contig":
        return F.log_softmax(torch.randn(N, C, dtype=dtype, device=device), dim=1)
    # Non-contiguous (N, C): transpose of a contiguous (C, N) tensor, so the
    # dispatch path exercises its .contiguous() copy fallback.
    return F.log_softmax(torch.randn(C, N, dtype=dtype, device=device), dim=0).t()


def warmup(fn, *args, **kwargs):
    for _ in range(10):
        try:
            fn(*args, **kwargs)
        except Exception:
            pass


def hdr(title):
    print(f"\n{'=' * 86}")
    print(f"  {title}")
    print(f"{'=' * 86}")
    print(
        f"  {'(N,C)':<18} {'dtype':<10} {'red':<8} {'layout':<10}"
        f" {'mean us':>10} {'median us':>10}"
    )
    print(f"  {'-' * 18} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")


def row(cfg, dtype, red, layout, m):
    print(
        f"  {str(cfg):<18} {str(dtype).split('.')[-1]:<10} {red:<8} {layout:<10}"
        f" {m.mean * 1e6:>10.2f} {m.median * 1e6:>10.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    _ = torch.randn(1024, device=device).sum()
    synchronize(device)

    hdr(f"NLLLoss (blocked_autorange) on {device}")
    for cfg, dtype, red, layout in itertools.product(
        CONFIGS, DTYPES, REDUCTIONS, LAYOUTS
    ):
        N, C = cfg
        lp = make_log_prob(N, C, dtype, device, layout)
        tgt = torch.randint(0, C, (N,), device=device)
        try:
            warmup(F.nll_loss, lp, tgt, reduction=red)
            synchronize(device)
            m = Timer(
                stmt="F.nll_loss(lp, tgt, reduction=red)",
                globals={"F": F, "lp": lp, "tgt": tgt, "red": red},
            ).blocked_autorange(min_run_time=2.0)
            row(cfg, dtype, red, layout, m)
        except Exception as e:
            print(f"  SKIP {cfg} {dtype} {red} {layout}: {e}")

    print()


if __name__ == "__main__":
    main()
