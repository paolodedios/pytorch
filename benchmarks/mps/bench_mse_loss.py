"""mse_loss: fused square-and-reduce kernel timing.

Measures F.mse_loss on the current build. The fused kernel computes (a-b)^2 and
reduces in one pass (no materialized squared-diff tensor). For the A/B vs the
MPSGraph path, run this on a parent (pre-migration) checkout for the baseline and
on this checkout for the fused kernel; the PR body carries that comparison.
Methodology: amortized many-iter / one-sync (dodges the MPS sync floor).

Defaults to the MPS device (this kernel's target) but takes --device, so the
same script profiles CUDA/CPU too. Covers contiguous and non-contiguous inputs.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F


DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
SHAPES = [
    ("8x2048x4096", (8, 2048, 4096)),
    ("1024x1024", (1024, 1024)),
    ("4x65536", (4, 65536)),
    ("65536x128", (65536, 128)),
    ("256x1024", (256, 1024)),
]
ITERS = int(os.environ.get("ITERS", "200"))


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(device, shape, dt, reduction, layout, backward):
    x = torch.randn(shape, device=device, dtype=dt)
    t = torch.randn(shape, device=device, dtype=dt)
    if layout == "noncontig" and x.dim() >= 2:
        # Transpose last two dims: non-contiguous input exercises the
        # dispatch's .contiguous() fallback.
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        t = t.transpose(-1, -2).contiguous().transpose(-1, -2)
    x.requires_grad_(backward)
    for _ in range(20):
        loss = F.mse_loss(x, t, reduction=reduction)
        if backward:
            x.grad = None
            loss.backward()
    synchronize(device)
    s = time.perf_counter()
    for _ in range(ITERS):
        loss = F.mse_loss(x, t, reduction=reduction)
        if backward:
            x.grad = None
            loss.backward()
    synchronize(device)
    return round((time.perf_counter() - s) / ITERS * 1e6, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    out = {}
    for nm, sh in SHAPES:
        for dn, dt in DTYPES.items():
            for red in ("mean", "sum"):
                for layout in ("contig", "noncontig"):
                    if layout == "noncontig" and len(sh) < 2:
                        continue
                    out[f"{nm}|{dn}|{red}|{layout}"] = bench(
                        device, sh, dt, red, layout, False
                    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
