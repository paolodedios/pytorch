"""
bench_loss_ops.py — benchmark Metal vs MPSGraph loss kernels on MPS.

Usage:
    python benchmarks/mps/bench_loss_ops.py

Prints mean±std throughput for each op across reduction modes and dtypes.
Uses torch.utils.benchmark.Timer with blocked_autorange per memory guidelines.
"""
import itertools
import torch
import torch.utils.benchmark as benchmark

DEVICE = "mps"
WARMUP = 30
MIN_RUN_TIME = 2.0  # seconds per cell

SHAPES = [(1024,), (256 * 256,), (32, 1024, 1024)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]


def bench_one(label, stmt, setup, globals_dict):
    t = benchmark.Timer(
        stmt=stmt,
        setup=setup,
        globals=globals_dict,
        label=label,
        num_threads=1,
    )
    m = t.blocked_autorange(min_run_time=MIN_RUN_TIME)
    return m.mean * 1e6, m.iqr * 1e6  # µs


def run_mse(shape, dtype, reduction):
    N = 1
    for s in shape:
        N *= s
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)
    loss_fn = torch.nn.MSELoss(reduction=reduction)
    g = {"loss_fn": loss_fn, "x": x, "y": y}
    mean_us, iqr = bench_one(
        f"MSELoss {shape} {dtype} {reduction}",
        "loss_fn(x, y).sum() if loss_fn.reduction == 'none' else loss_fn(x, y)",
        "",
        g,
    )
    return mean_us, iqr


def run_smooth_l1(shape, dtype, reduction, beta=1.0):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)
    loss_fn = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
    g = {"loss_fn": loss_fn, "x": x, "y": y}
    mean_us, iqr = bench_one(
        f"SmoothL1 {shape} {dtype} {reduction}",
        "loss_fn(x, y).sum() if loss_fn.reduction == 'none' else loss_fn(x, y)",
        "",
        g,
    )
    return mean_us, iqr


def run_huber(shape, dtype, reduction, delta=1.0):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y = torch.randn(shape, device=DEVICE, dtype=dtype)
    loss_fn = torch.nn.HuberLoss(reduction=reduction, delta=delta)
    g = {"loss_fn": loss_fn, "x": x, "y": y}
    mean_us, iqr = bench_one(
        f"HuberLoss {shape} {dtype} {reduction}",
        "loss_fn(x, y).sum() if loss_fn.reduction == 'none' else loss_fn(x, y)",
        "",
        g,
    )
    return mean_us, iqr


def run_bce(shape, dtype, reduction):
    x = torch.sigmoid(torch.randn(shape, device=DEVICE, dtype=dtype))
    y = torch.randint(0, 2, shape, device=DEVICE).to(dtype)
    loss_fn = torch.nn.BCELoss(reduction=reduction)
    g = {"loss_fn": loss_fn, "x": x, "y": y}
    mean_us, iqr = bench_one(
        f"BCELoss {shape} {dtype} {reduction}",
        "loss_fn(x, y).sum() if loss_fn.reduction == 'none' else loss_fn(x, y)",
        "",
        g,
    )
    return mean_us, iqr


def run_nll(N, C, dtype, reduction):
    x = torch.log_softmax(torch.randn(N, C, device=DEVICE, dtype=dtype), dim=1)
    t = torch.randint(0, C, (N,), device=DEVICE)
    loss_fn = torch.nn.NLLLoss(reduction=reduction)
    g = {"loss_fn": loss_fn, "x": x, "t": t}
    mean_us, iqr = bench_one(
        f"NLLLoss ({N},{C}) {dtype} {reduction}",
        "loss_fn(x, t).sum() if loss_fn.reduction == 'none' else loss_fn(x, t)",
        "",
        g,
    )
    return mean_us, iqr


def run_cross_entropy(N, C, dtype, reduction):
    x = torch.randn(N, C, device=DEVICE, dtype=dtype)
    t = torch.randint(0, C, (N,), device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    g = {"loss_fn": loss_fn, "x": x, "t": t}
    mean_us, iqr = bench_one(
        f"CrossEntropy ({N},{C}) {dtype} {reduction}",
        "loss_fn(x, t).sum() if loss_fn.reduction == 'none' else loss_fn(x, t)",
        "",
        g,
    )
    return mean_us, iqr


def header(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  {'Shape':<22} {'dtype':<10} {'reduction':<8} {'mean µs':>10} {'±iqr µs':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")


def row(shape, dtype, reduction, mean, iqr):
    print(f"  {str(shape):<22} {str(dtype).split('.')[-1]:<10} {reduction:<8} {mean:>10.1f} {iqr:>10.1f}")


if __name__ == "__main__":
    # Warmup
    _ = torch.randn(1024, device=DEVICE).sum()
    torch.mps.synchronize()

    POINTWISE_SHAPES = [(65536,), (1 << 20,), (32, 1024)]

    header("MSELoss (fwd)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        try:
            m, iqr = run_mse(shape, dtype, red)
            row(shape, dtype, red, m, iqr)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    header("SmoothL1Loss (fwd, beta=1.0)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        try:
            m, iqr = run_smooth_l1(shape, dtype, red)
            row(shape, dtype, red, m, iqr)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    header("HuberLoss (fwd, delta=1.0)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        try:
            m, iqr = run_huber(shape, dtype, red)
            row(shape, dtype, red, m, iqr)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    header("BCELoss (fwd)")
    for shape, dtype, red in itertools.product(POINTWISE_SHAPES, DTYPES, REDUCTIONS):
        try:
            m, iqr = run_bce(shape, dtype, red)
            row(shape, dtype, red, m, iqr)
        except Exception as e:
            print(f"  SKIP {shape} {dtype} {red}: {e}")

    NLL_CONFIGS = [(256, 10), (256, 1000), (1024, 10000)]
    header("NLLLoss (fwd)")
    for (N, C), dtype, red in itertools.product(NLL_CONFIGS, DTYPES, REDUCTIONS):
        try:
            m, iqr = run_nll(N, C, dtype, red)
            row(f"({N},{C})", dtype, red, m, iqr)
        except Exception as e:
            print(f"  SKIP ({N},{C}) {dtype} {red}: {e}")

    header("CrossEntropyLoss (fwd, fused CE)")
    for (N, C), dtype, red in itertools.product(NLL_CONFIGS, DTYPES, REDUCTIONS):
        try:
            m, iqr = run_cross_entropy(N, C, dtype, red)
            row(f"({N},{C})", dtype, red, m, iqr)
        except Exception as e:
            print(f"  SKIP ({N},{C}) {dtype} {red}: {e}")

    print()
