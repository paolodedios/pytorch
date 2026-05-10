"""
Throughput benchmark for MPS loss ops.

Runs N iterations with ONE sync at the end ??? measures amortized per-call cost
in a pipeline context (what matters for training), not isolated-call latency
which is dominated by command buffer commit overhead.

Run: python tmp_bench_loss_throughput.py [metal|mpsgraph]
"""
import time
import itertools
import torch
import torch.nn.functional as F

DEVICE = "mps"
N_WARMUP = 100
N_ITERS = 2000


def bench_fn(fn, *args, **kwargs):
    for _ in range(N_WARMUP):
        fn(*args, **kwargs)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn(*args, **kwargs)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / N_ITERS * 1e6  # ??s


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]

POINTWISE_SHAPES = [(1024,), (65536,), (1 << 20,), (32, 1024)]
NLL_CONFIGS = [(256, 10), (256, 1000), (1024, 10000)]


def header(title):
    print(f"\n{'???'*68}")
    print(f"  {title}")
    print(f"{'???'*68}")
    print(f"  {'Shape':<22} {'dtype':<10} {'reduction':<8} {'??s/call':>10}")
    print(f"  {'???'*22} {'???'*10} {'???'*8} {'???'*10}")


def row(shape, dtype, reduction, us):
    print(f"  {str(shape):<22} {str(dtype).split('.')[-1]:<10} {reduction:<8} {us:>10.2f}")


if __name__ == "__main__":
    _ = torch.randn(1024, device=DEVICE).sum()
    torch.mps.synchronize()

    header("MSELoss (fwd throughput)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        x = torch.randn(shape, device=DEVICE, dtype=dtype)
        y = torch.randn(shape, device=DEVICE, dtype=dtype)
        try:
            us = bench_fn(F.mse_loss, x, y, reduction=red)
            row(shape, dtype, red, us)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    header("SmoothL1Loss (fwd throughput, beta=1.0)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        x = torch.randn(shape, device=DEVICE, dtype=dtype)
        y = torch.randn(shape, device=DEVICE, dtype=dtype)
        try:
            us = bench_fn(F.smooth_l1_loss, x, y, reduction=red)
            row(shape, dtype, red, us)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    header("HuberLoss (fwd throughput, delta=1.0)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        x = torch.randn(shape, device=DEVICE, dtype=dtype)
        y = torch.randn(shape, device=DEVICE, dtype=dtype)
        try:
            us = bench_fn(F.huber_loss, x, y, reduction=red)
            row(shape, dtype, red, us)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    header("BCELoss (fwd throughput)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        x = torch.sigmoid(torch.randn(shape, device=DEVICE, dtype=dtype))
        y = torch.randint(0, 2, shape, device=DEVICE).to(dtype)
        try:
            us = bench_fn(F.binary_cross_entropy, x, y, reduction=red)
            row(shape, dtype, red, us)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    header("NLLLoss (fwd throughput)")
    for (N, C), dtype, red in itertools.product(NLL_CONFIGS, DTYPES, REDUCTIONS):
        x = torch.log_softmax(torch.randn(N, C, device=DEVICE, dtype=dtype), dim=1)
        t = torch.randint(0, C, (N,), device=DEVICE)
        try:
            us = bench_fn(F.nll_loss, x, t, reduction=red)
            row(f"({N},{C})", dtype, red, us)
        except Exception as e:
            print(f"  SKIP ({N},{C}) {dtype} {red}: {e}")

    header("CrossEntropyLoss (fused fwd throughput)")
    for (N, C), dtype, red in itertools.product(NLL_CONFIGS, DTYPES, REDUCTIONS):
        x = torch.randn(N, C, device=DEVICE, dtype=dtype)
        t = torch.randint(0, C, (N,), device=DEVICE)
        try:
            us = bench_fn(F.cross_entropy, x, t, reduction=red)
            row(f"({N},{C})", dtype, red, us)
        except Exception as e:
            print(f"  SKIP ({N},{C}) {dtype} {red}: {e}")

    print()
