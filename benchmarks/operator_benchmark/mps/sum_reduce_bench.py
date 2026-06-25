"""Benchmark for the consolidated sum / mean / nansum / count_nonzero reductions,
which now route through the shared two-pass value_reduction<SumOp> Metal kernel
(the same kernel min/max/all/any already use) instead of a separate sum_reduction
kernel.

Run this on a build WITH the change and on the parent commit (without it) and
compare. The consolidation is behavior-preserving, so the expectation is parity
(no regression), not a speedup. amax / amin are included as an unchanged-kernel
control: their kernel is identical on both builds, so any apparent amax/amin
delta is pure machine drift and can be divided out of the sum delta
(corrected = (mine_sum / base_sum) / (mine_amax / base_amax)).

Methodology note: the per-shape MPS bench has high run-to-run variance and is
sensitive to machine load/thermal. Run this script on each build back-to-back on
an otherwise-idle machine and compare medians; do not compare a fresh run against
a stored baseline taken under different load.
"""

import torch
import torch.utils.benchmark as benchmark


# Large shapes so the kernel cost dominates the ~110us MPS dispatch/sync floor.
SHAPES = [
    ((1 << 24,), None),
    ((4096, 4096), 1),
    ((4096, 4096), 0),
    ((1024, 16384), 1),
    ((8192, 1024), 0),
    ((1_000_000, 8), 1),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# sum/mean/nansum migrate onto value_reduction<SumOp>; amax/amin are the
# unchanged-kernel control.
OPS = {
    "sum": torch.sum,
    "mean": torch.mean,
    "nansum": torch.nansum,
    "amax": torch.amax,
    "amin": torch.amin,
}


def steady_state():
    print("# steady-state reduction (median us, warm)")
    for name, fn in OPS.items():
        for dtype in DTYPES:
            for shape, dim in SHAPES:
                x = torch.randn(*shape, device="mps").to(dtype)
                torch.mps.synchronize()
                if dim is None:
                    t = benchmark.Timer(
                        stmt="fn(x); torch.mps.synchronize()",
                        globals={"fn": fn, "x": x, "torch": torch},
                    )
                else:
                    t = benchmark.Timer(
                        stmt="fn(x, dim=dim); torch.mps.synchronize()",
                        globals={"fn": fn, "x": x, "dim": dim, "torch": torch},
                    )
                # Median of 5 trials: a single blocked_autorange has up to ~2x
                # run-to-run spikes at large float32 shapes (verified -- even the
                # unchanged amax/amin kernel spikes that much), so a 5-trial
                # median is required to compare builds meaningfully.
                trials = sorted(
                    t.blocked_autorange(min_run_time=0.3).median for _ in range(5)
                )
                us = trials[2] * 1e6
                dt = str(dtype).replace("torch.", "")
                print(f"  {name:7} {dt:9} {str(shape):16} dim={dim}: {us:9.1f} us")
                del x
                torch.mps.empty_cache()


if __name__ == "__main__":
    steady_state()
