"""
Benchmark: eigh-based `matrix_sqrth` vs proposed Schur-based `matrix_sqrt`.

Run:
    ```shell
    .venv/bin/python -m benchmarks.linalg.matrix_sqrt_benchmark --out results.md
    ```
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

import torch
import torch.utils.benchmark as benchmark
import scipy.linalg as _scipy_linalg

DEFAULT_THREADS = torch.get_num_threads()

SIZES = [8, 32, 128, 512, 1024]
BATCHES = [1, 16]
DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "cfloat": torch.complex64,
    "cdouble": torch.complex128,
}

MIN_RUN_TIME = 0.3
ACCURACY_SEEDS = [0, 1, 2]
# scipy.linalg.sqrtm is O(n^3) in Python; cap to a representative subset for the oracle check.
SCIPY_MAX_N = 256
# Lifts B @ B^H off the PSD boundary so decompositions are well-conditioned.
PSD_EPS = 1e-3

EIGH_LABEL = "matrix_sqrth (eigh)"
SCHUR_LABEL = "matrix_sqrt (schur)"


def make_psd(n, batch, dtype, seed=42):
    rng = np.random.default_rng(seed)
    shape = (batch, n, n) if batch > 1 else (n, n)
    is_complex = dtype in (torch.complex64, torch.complex128)
    real_np = np.float64 if dtype in (torch.float64, torch.complex128) else np.float32

    if is_complex:
        real = rng.standard_normal(shape).astype(real_np)
        imag = rng.standard_normal(shape).astype(real_np)
        bt = torch.from_numpy(real + 1j * imag).to(dtype)
    else:
        bt = torch.from_numpy(rng.standard_normal(shape).astype(real_np)).to(dtype)

    a = bt @ bt.mH + PSD_EPS * torch.eye(n, dtype=dtype)

    return 0.5 * (a + a.mH)


def time_method(fn, a, nthreads, min_run_time):
    timer = benchmark.Timer(
        stmt="fn(a)",
        globals={"fn": fn, "a": a},
        num_threads=nthreads,
    )
    m = timer.blocked_autorange(min_run_time=min_run_time)
    rel_iqr = (m.iqr / m.median) if m.median > 0 else 0.0

    return {"median": m.median, "mean": m.mean, "rel_iqr": rel_iqr}


def fro(x):
    return torch.linalg.matrix_norm(x)


def scipy_rel_err(a, x_schur):
    """
    Returns worst-case relative Frobenius error vs scipy.linalg.sqrtm over the batch.
    """
    a_np = a.detach().cpu().numpy().reshape(-1, a.shape[-1], a.shape[-1])
    x_np = x_schur.detach().cpu().numpy().reshape(-1, a.shape[-1], a.shape[-1])
    worst = 0.0

    for ai, xi in zip(a_np, x_np):
        ref = np.asarray(_scipy_linalg.sqrtm(ai))
        den = np.linalg.norm(ai)

        if den > 0:
            worst = max(worst, float(np.linalg.norm(xi - ref) / den))
    return worst


def accuracy_probe(methods, n, batch, dtype, seeds):
    rt_eigh = rt_schur = agree = 0.0
    sci = None
    do_scipy = n <= SCIPY_MAX_N
    for seed in seeds:
        a = make_psd(n, batch, dtype, seed=seed)
        na = fro(a)
        x_eigh = methods[EIGH_LABEL](a)
        x_schur = methods[SCHUR_LABEL](a)
        rt_eigh = max(rt_eigh, float((fro(x_eigh @ x_eigh - a) / na).max().item()))
        rt_schur = max(rt_schur, float((fro(x_schur @ x_schur - a) / na).max().item()))
        agree = max(agree, float((fro(x_eigh - x_schur) / na).max().item()))
        if do_scipy:
            err = scipy_rel_err(a, x_schur)
            sci = err if sci is None else max(sci, err)
    return {"rt_eigh": rt_eigh, "rt_schur": rt_schur, "agree": agree, "scipy": sci}


def burn_in(seconds=10.0):
    # Drive the chip to a stable clock before timing.
    a = make_psd(256, 1, torch.float64)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        torch.linalg.eigh(a)


def _sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name]).decode().strip()
    except Exception:
        return None


def env_header():
    try:
        macos = subprocess.check_output(["sw_vers", "-productVersion"]).decode().strip()
    except Exception:
        macos = platform.mac_ver()[0] or "unknown"

    cpu = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    pcores = _sysctl("hw.perflevel0.physicalcpu")
    ecores = _sysctl("hw.perflevel1.physicalcpu")
    if pcores and ecores:
        topology = f"{pcores} performance + {ecores} efficiency cores"
    else:
        topology = f"{os.cpu_count()} logical cores"
    cfg = torch.__config__.show()
    blas_m = re.search(r"BLAS_INFO=(\w+)", cfg)
    lapack_m = re.search(r"LAPACK_INFO=(\w+)", cfg)
    blas = "{} (BLAS), {} (LAPACK)".format(
        blas_m.group(1) if blas_m else "unknown",
        lapack_m.group(1) if lapack_m else "unknown",
    )
    return {
        "macOS": macos,
        "cpu": cpu,
        "topology": topology,
        "Python": sys.version.split()[0],
        "PyTorch": torch.__version__,
        "git": torch.version.git_version,
        "blas": blas,
        "parallel_info": torch.__config__.parallel_info().strip(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "unset"),
        "nice": os.nice(0),
        "scipy": _scipy_linalg.__name__,
    }


def run(out_path, do_burn_in=True, run_label=""):
    methods = {
        EIGH_LABEL: torch.linalg.matrix_sqrth,
        SCHUR_LABEL: torch.linalg.matrix_sqrt,
    }
    header = env_header()
    print(f"default threads: {DEFAULT_THREADS}\n")

    if do_burn_in:
        print("burn-in (10s)...")
        burn_in()

    thread_passes = [("threads=1", 1), ("threads=N", DEFAULT_THREADS)]
    # results[batch][(n, dtype)][method][pass_label] = timing dict
    results = {b: {} for b in BATCHES}
    accuracy = {}

    for batch in BATCHES:
        for dtype_name, dtype in DTYPES.items():
            for n in SIZES:
                # Cap the heaviest cell so total runtime stays reasonable.
                if n == 1024 and batch > 1:
                    continue
                a = make_psd(n, batch, dtype)
                key = (n, dtype_name)
                results[batch].setdefault(key, {})
                for method_label, fn in methods.items():
                    results[batch][key].setdefault(method_label, {})
                    for pass_label, nthreads in thread_passes:
                        print(
                            f"n={n}, batch={batch}, dtype={dtype_name}, "
                            f"method={method_label}, {pass_label}"
                        )
                        results[batch][key][method_label][pass_label] = time_method(
                            fn, a, nthreads, MIN_RUN_TIME
                        )
                accuracy[(batch, n, dtype_name)] = accuracy_probe(
                    methods, n, batch, dtype, ACCURACY_SEEDS
                )

    write_results(out_path, header, methods, results, accuracy, run_label)


def fmt_us(seconds):
    return f"{seconds * 1e6:.1f}"


def fmt_pct(frac):
    return f"{frac * 100:.1f}%"


def write_results(out_path, header, methods, results, accuracy, label=""):
    lines = []

    if label:
        lines.append(f"# {label}\n")

    lines.append("## Environment\n")
    lines.append("| | |")
    lines.append("|---|---|")
    for k, v in header.items():
        v_str = str(v).replace("\n", " / ")
        lines.append(f"| {k} | {v_str} |")
    lines.append("")

    for batch in BATCHES:
        title = "single matrix" if batch == 1 else f"batch size {batch}"
        lines.append(f"## Timing -- {title}\n")
        lines.append(
            "| dtype | n | method | threads=1 median | threads=1 IQR% | "
            "threads=N median | threads=N IQR% |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        for dtype_name in DTYPES:
            for n in SIZES:
                key = (n, dtype_name)
                if key not in results[batch]:
                    continue
                for method_label in methods:
                    cell = results[batch][key][method_label]
                    t1, tn = cell["threads=1"], cell["threads=N"]
                    lines.append(
                        f"| {dtype_name} | {n} | {method_label} | "
                        f"{fmt_us(t1['median'])} | {fmt_pct(t1['rel_iqr'])} | "
                        f"{fmt_us(tn['median'])} | {fmt_pct(tn['rel_iqr'])} |"
                    )
        lines.append("")

    lines.append("## Speed ratio (Schur / eigh), threads=N median\n")
    lines.append(
        "Ratio > 1 means the Schur op is slower than eigh. Computed from the "
        "threads=N median times.\n"
    )
    lines.append("| dtype | n | batch | eigh us | schur us | schur/eigh |")
    lines.append("|---|---|---|---|---|---|")

    for batch in BATCHES:
        for dtype_name in DTYPES:
            for n in SIZES:
                key = (n, dtype_name)
                if key not in results[batch]:
                    continue
                e = results[batch][key][EIGH_LABEL]["threads=N"]["median"]
                s = results[batch][key][SCHUR_LABEL]["threads=N"]["median"]
                ratio = s / e if e > 0 else float("nan")
                lines.append(
                    f"| {dtype_name} | {n} | {batch} | {fmt_us(e)} | {fmt_us(s)} | "
                    f"{ratio:.2f}x |"
                )
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="matrix_sqrt_results.md",
        help="path for the generated results markdown",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fast sanity sweep (small sizes, short timing budget, no burn-in)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="scenario name recorded in the report header (e.g. '1-core', '8P-cores')",
    )
    args = parser.parse_args()

    if args.quick:
        global SIZES, BATCHES, MIN_RUN_TIME, ACCURACY_SEEDS
        SIZES = [8, 32]
        BATCHES = [1]
        MIN_RUN_TIME = 0.05
        ACCURACY_SEEDS = [0]

    run(args.out, do_burn_in=not args.quick, run_label=args.label)


if __name__ == "__main__":
    main()
