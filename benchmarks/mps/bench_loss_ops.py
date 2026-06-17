"""
Benchmark for the smooth_l1_loss and huber_loss ops.

Defaults to the MPS device (this kernel's target) but works on any device via
--device, so the same script can profile the Metal path against CUDA/CPU.
Uses torch.utils.benchmark.Timer.blocked_autorange -- no hand-rolled timing.

For reduction=none, fwd+bwd uses .backward(rand_weights) to model the real
use case (per-sample importance weighting). reduction=mean/sum use .sum().backward().

Covers smooth_l1_loss + huber_loss (forward + backward), contiguous and
non-contiguous inputs, over float32/float16/bfloat16 and standard + LLM-scale
activation shapes.

Run: python benchmarks/mps/bench_loss_ops.py [--device mps]
"""

import argparse
import itertools

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


MIN_RUN = 1.0  # seconds per config for blocked_autorange


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]
LAYOUTS = ["contig", "noncontig"]

# Pointwise shapes: standard small/medium/large + LLM-scale activations.
POINTWISE_SHAPES = [
    (4096,),  # 1D medium
    (1 << 20,),  # 1D large (1M)
    (32, 4096),  # batch=32, hidden=4096 (LLaMA hidden)
    (16, 512, 4096),  # batch=16, seq=512, hidden=4096 (LLM activations)
    (8, 2048, 4096),  # batch=8, seq=2048, hidden=4096
]

OPS = [("smooth_l1_loss", F.smooth_l1_loss), ("huber_loss", F.huber_loss)]


def hdr(title):
    print(f"\n{'=' * 104}")
    print(f"  {title}")
    print(f"{'=' * 104}")
    print(
        f"  {'shape':<28} {'dtype':<10} {'red':<6} {'layout':<10}"
        f" {'fwd median us':>15} {'fwd+bwd median us':>19}"
    )
    print(f"  {'-' * 28} {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 15} {'-' * 19}")


def row(shape, dtype, red, layout, fwd, fwdbwd):
    dt = str(dtype).split(".")[-1]
    bwd_s = f"{fwdbwd:>19.2f}" if fwdbwd is not None else f"{'-':>19}"
    print(f"  {str(shape):<28} {dt:<10} {red:<6} {layout:<10} {fwd:>15.2f} {bwd_s}")


def timed(stmt, g):
    try:
        return (
            Timer(stmt=stmt, globals=g).blocked_autorange(min_run_time=MIN_RUN).median
            * 1e6
        )
    except Exception:
        return None


def make_noncontig(t):
    # Transpose the last two dims so the input is non-contiguous; the dispatch
    # then exercises its .contiguous() fallback. 1D shapes have no transpose.
    if t.dim() < 2:
        return t
    return t.transpose(-1, -2).contiguous().transpose(-1, -2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    for _ in range(30):
        torch.randn(1024, device=device).sum()
    synchronize(device)

    for op_name, op in OPS:
        hdr(f"{op_name} -- forward | forward+backward on {device}")
        for shape, dtype, red, layout in itertools.product(
            POINTWISE_SHAPES, DTYPES, REDUCTIONS, LAYOUTS
        ):
            if layout == "noncontig" and len(shape) < 2:
                continue
            x = torch.randn(shape, device=device, dtype=dtype)
            y = torch.randn(shape, device=device, dtype=dtype)
            if layout == "noncontig":
                x, y = make_noncontig(x), make_noncontig(y)
            g = dict(
                op=op,
                x=x,
                y=y,
                red=red,
                xg=x.detach().requires_grad_(True),
                w=torch.rand_like(x),
            )
            fwd = timed("op(x, y, reduction=red)", g)
            if fwd is None:
                continue
            bwd_stmt = (
                "xg.grad=None; op(xg, y, reduction='none').backward(w)"
                if red == "none"
                else "xg.grad=None; op(xg, y, reduction=red).sum().backward()"
            )
            row(shape, dtype, red, layout, fwd, timed(bwd_stmt, g))

    print()


if __name__ == "__main__":
    main()
