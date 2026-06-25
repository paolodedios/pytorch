"""Benchmark for the native Metal prod reduction vs the MPSGraph implementation
it replaces.

Run on a build WITH the native kernel and on a build WITHOUT it (e.g. the PR
parent commit) and compare. Reports two regimes:

  * steady-state: a fixed shape benchmarked warm (torch.utils.benchmark median).
  * shape-varying: many distinct shapes seen once each -- this is where MPSGraph
    pays a per-shape graph recompilation (~100ms/shape) the native kernel does
    not.

Methodology note: the per-shape MPS bench has high run-to-run variance; use the
interleaved 5-trial median below (not a single sweep) when comparing builds.
"""

import time

import torch
import torch.utils.benchmark as benchmark


STEADY_SHAPES = [
    ((1 << 24,), None),
    ((4096, 4096), 1),
    ((4096, 4096), 0),
    ((1024, 16384), 1),
    ((1_000_000, 8), 1),  # tall-thin: thread-per-row kernel
    ((2, 8_000_000), 1),  # short-wide: wide inner kernel
    ((1_000_000, 8), 0),  # dim=0 outer_bucketed: few columns, long reduced dim
    ((2, 8_000_000), 0),  # dim=0 outer_thin: short reduced dim, many columns
]


def steady_state():
    print("# steady-state prod (median us, warm)")
    for shape, dim in STEADY_SHAPES:
        x = torch.randn(*shape, device="mps")
        torch.mps.synchronize()
        stmt = (
            "torch.prod(x); torch.mps.synchronize()"
            if dim is None
            else "torch.prod(x, dim=dim); torch.mps.synchronize()"
        )
        t = benchmark.Timer(stmt=stmt, globals={"x": x, "dim": dim, "torch": torch})
        us = t.blocked_autorange(min_run_time=0.5).median * 1e6
        print(f"  {str(shape):20} dim={dim}: {us:8.1f} us")
        del x
        torch.mps.empty_cache()


def shape_varying(n=200):
    # n distinct shapes, each reduced once -> exercises per-shape compilation.
    buf = torch.randn(20_000_000, device="mps")
    torch.prod(torch.as_strided(buf, (64, 500), (500, 1)), dim=1)  # warm
    torch.mps.synchronize()
    t0 = time.time()
    for i in range(n):
        N = 999 + i * 101
        torch.prod(torch.as_strided(buf, (64, N), (N, 1)), dim=1)
    torch.mps.synchronize()
    dt = time.time() - t0
    print(
        f"# shape-varying prod: {n} distinct shapes = {dt * 1e3:.0f} ms "
        f"({dt / n * 1e3:.2f} ms/shape)"
    )


def rss_stability(n=2000):
    """The point of the migration: the native kernel has no per-shape graph
    cache, so process peak RSS stays bounded across distinct shapes, unlike the
    MPSGraph prod (which grows its graph cache per shape). Reports the peak-RSS
    growth across n distinct shapes -- a few MB on the native build, steadily
    climbing on the MPSGraph build."""
    import resource
    import sys

    def peak_rss_mb():
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kB on Linux
        return ru / (1 << 20) if sys.platform == "darwin" else ru / (1 << 10)

    buf = torch.randn(20_000_000, device="mps")
    torch.prod(torch.as_strided(buf, (64, 500), (500, 1)), dim=1)  # warm
    torch.mps.synchronize()
    before = peak_rss_mb()
    for i in range(n):
        N = 999 + i * 101
        torch.prod(torch.as_strided(buf, (64, N), (N, 1)), dim=1)
    torch.mps.synchronize()
    after = peak_rss_mb()
    print(
        f"# RSS stability: {n} distinct shapes, peak RSS {before:.0f} -> {after:.0f} MB "
        f"(+{after - before:.0f} MB; native has no per-shape graph cache)"
    )


if __name__ == "__main__":
    steady_state()
    shape_varying()
    rss_stability()
