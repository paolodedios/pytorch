"""Benchmark CPU->MPS tensor copy: zero-copy fast path vs blit baseline.

Fast path: contiguous same-dtype tensors — [destBuffer contents] + memcpy, no Metal command.
Blit path: non-contiguous tensors, dtype casts, discrete GPU (fallback).

Baseline numbers below were measured on the parent commit (blit-only) with the same
warmup/iters settings. Run this script after building with the fast-path patch applied.

Usage:
    PYTHONPATH=~/pytorch python benchmarks/bench_mps_copy_to_mps.py
"""
import math
import time

import torch

def bench(fn, warmup=30, iters=100):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    mu = sum(times) / len(times)
    sigma = math.sqrt(sum((t - mu)**2 for t in times) / len(times))
    return mu, sigma

# Baseline (blit path) numbers measured on parent commit (Apple M1 Max):
BASELINE_US = {
    "scalar (1 elem)": 270,
    "1 KB float32": 110,
    "64 KB float32": 102,
    "1 MB float32": 311,
    "16 MB float32": 465,
    "1 MB float16": 319,
    "1 MB bfloat16": 319, # similar to float16
    "1 MB int32": 116,
    "1 MB int64": 117,
    "1 MB bool": 110,
}

def speedup_str(label, after_us):
    base = BASELINE_US.get(label)
    if base is None:
        return ""
    return f"  [{base:.0f} µs → {after_us:.1f} µs, {base/after_us:.1f}x]"

def run_size(shape, dtype=torch.float32, label=""):
    hi = 2 if dtype == torch.bool else 10
    cpu_t = torch.randn(shape, dtype=dtype) if dtype.is_floating_point else torch.randint(0, hi, shape, dtype=dtype)
    nbytes = cpu_t.nbytes

    # contiguous same-dtype: fast path
    mu, sigma = bench(lambda: cpu_t.to("mps"))
    bw = nbytes / (mu * 1e-6) / 1e9
    sp = speedup_str(label, mu)
    print(f"  {label:30s}  {mu:8.2f} ± {sigma:5.2f} µs  ({bw:.1f} GB/s){sp}")

    # non-contiguous CPU tensor: goes through clone() first then fast path on contiguous result
    cpu_nc = cpu_t[::2] if cpu_t.dim() == 1 else cpu_t[:, ::2]
    mu_nc, sigma_nc = bench(lambda: cpu_nc.to("mps"))
    print(f"  {label+' (strided)':30s}  {mu_nc:8.2f} ± {sigma_nc:5.2f} µs  (clone+fast path)")

print(f"PyTorch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
print()

sizes = [
    ((1,), "scalar (1 elem)"),
    ((256,), "1 KB float32"),
    ((16384,), "64 KB float32"),
    ((262144,), "1 MB float32"),
    ((1048576,), "4 MB float32"),
    ((4194304,), "16 MB float32"),
    ((128, 128), "128x128 float32"),
    ((1024, 1024), "1024x1024 float32"),
]

print("CPU → MPS copy benchmark (mean ± σ of 100 runs, 30 warmup):")
print("  [baseline → fast path, speedup] measured on Apple M1 Max")
print("-" * 90)
for shape, label in sizes:
    run_size(shape, label=label)

print()
print("dtype coverage (1 MB tensors, contiguous):")
print("-" * 90)
for dtype, label in [
    (torch.float32, "float32"),
    (torch.float16, "float16"),
    (torch.bfloat16, "bfloat16"),
    (torch.int32, "int32"),
    (torch.int64, "int64"),
    (torch.bool, "bool"),
]:
    n = 1048576 // torch.tensor([], dtype=dtype).element_size()
    run_size((n,), dtype=dtype, label=f"1 MB {label}")
