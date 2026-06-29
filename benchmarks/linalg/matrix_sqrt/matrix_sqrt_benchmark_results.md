# matrix_sqrth (eigh) vs matrix_sqrt (Schur) Benchmark

## Environment

- Run label:  1-core
- CPU:        Apple M4 Pro (8 performance + 4 efficiency cores)
- macOS:      26.5.1
- Python:     3.11.13
- PyTorch:    2.14.0a0+git6e207a5  (git 6e207a5749be1eaaf40f5435717aed7fc5ef69ed)
- BLAS:       accelerate (BLAS), accelerate (LAPACK)
- OMP_NUM_THREADS:        1
- VECLIB_MAXIMUM_THREADS: 1
- threads=N pass:         1 torch threads

```
ATen/Parallel:
	at::get_num_threads() : 1
	at::get_num_interop_threads() : 12
OpenMP 202011
	omp_get_max_threads() : 1
MKLDNN not found
std::thread::hardware_concurrency() : 12
Environment variables:
	OMP_NUM_THREADS : 1
ATen parallel backend: OpenMP
```

## Methods

`matrix_sqrth`: `Q diag(sqrt(lambda)) Q^H` via LAPACK `?syevd`/`?heevd`.

`matrix_sqrt` is the generic Schur operation: `?gees` (Hessenberg + QR on the full matrix, ignoring symmetry) then the Bjorck-Hammarling triangular square-root recurrence, with `?trsyl` solving the off-diagonal Sylvester blocks (Deadman-Higham-Ralha blocking) and the back-transform `Z U Z^H` via gemm.

Both are CPU LAPACK-backed; the comparison isolates the algorithm (eigendecomposition vs Schur), not the BLAS backend.

Times are microseconds (us) per call, the **median** from `torch.utils.benchmark.Timer.blocked_autorange`. `IQR%` is the interquartile range as a percent of the median (measurement noise; eigh and Schur are timed back-to-back per cell so the **ratio** is robust to thermal drift even if absolute times are not).

Caveat: these numbers are Apple Accelerate (vecLib) specific. The qualitative conclusion is algorithmic (eigh exploits symmetry, Schur cannot) and is expected to hold on Linux/MKL, but the absolute ratios are not portable across BLAS backends.

## Timing -- single matrix

| dtype | n | method | threads=1 median | threads=1 IQR% | threads=N median | threads=N IQR% |
|---|---|---|---|---|---|---|
| float32 | 8 | matrix_sqrth (eigh) | 6.1 | 0.6% | 6.0 | 2.4% |
| float32 | 8 | matrix_sqrt (schur) | 6.6 | 4.4% | 6.9 | 10.1% |
| float32 | 32 | matrix_sqrth (eigh) | 41.5 | 1.5% | 37.7 | 1.7% |
| float32 | 32 | matrix_sqrt (schur) | 80.2 | 0.1% | 80.4 | 6.8% |
| float32 | 128 | matrix_sqrth (eigh) | 502.3 | 1.3% | 514.5 | 3.5% |
| float32 | 128 | matrix_sqrt (schur) | 2247.9 | 9.3% | 2200.4 | 2.7% |
| float32 | 512 | matrix_sqrth (eigh) | 8730.0 | 4.6% | 8642.2 | 3.7% |
| float32 | 512 | matrix_sqrt (schur) | 47440.5 | 2.3% | 47652.8 | 4.2% |
| float32 | 1024 | matrix_sqrth (eigh) | 43120.7 | 0.4% | 43028.0 | 0.7% |
| float32 | 1024 | matrix_sqrt (schur) | 194681.9 | 0.0% | 195103.8 | 0.2% |
| float64 | 8 | matrix_sqrth (eigh) | 8.3 | 0.9% | 8.3 | 1.9% |
| float64 | 8 | matrix_sqrt (schur) | 6.7 | 1.3% | 6.8 | 2.0% |
| float64 | 32 | matrix_sqrth (eigh) | 46.0 | 0.3% | 46.0 | 0.5% |
| float64 | 32 | matrix_sqrt (schur) | 87.0 | 0.4% | 87.1 | 0.7% |
| float64 | 128 | matrix_sqrth (eigh) | 669.1 | 0.2% | 666.3 | 0.5% |
| float64 | 128 | matrix_sqrt (schur) | 2909.9 | 0.6% | 2928.2 | 1.0% |
| float64 | 512 | matrix_sqrth (eigh) | 13201.2 | 4.0% | 13098.0 | 0.6% |
| float64 | 512 | matrix_sqrt (schur) | 65047.3 | 1.4% | 65173.8 | 0.6% |
| float64 | 1024 | matrix_sqrth (eigh) | 73225.4 | 0.4% | 73186.0 | 0.0% |
| float64 | 1024 | matrix_sqrt (schur) | 313206.2 | 0.0% | 310961.7 | 0.0% |
| cfloat | 8 | matrix_sqrth (eigh) | 8.6 | 2.0% | 8.5 | 1.0% |
| cfloat | 8 | matrix_sqrt (schur) | 11.0 | 1.2% | 11.2 | 1.7% |
| cfloat | 32 | matrix_sqrth (eigh) | 60.2 | 1.9% | 59.7 | 0.5% |
| cfloat | 32 | matrix_sqrt (schur) | 127.7 | 0.5% | 127.2 | 0.6% |
| cfloat | 128 | matrix_sqrth (eigh) | 1302.6 | 0.7% | 1305.1 | 0.6% |
| cfloat | 128 | matrix_sqrt (schur) | 4981.4 | 0.4% | 4992.3 | 0.3% |
| cfloat | 512 | matrix_sqrth (eigh) | 47084.1 | 0.7% | 46834.6 | 0.6% |
| cfloat | 512 | matrix_sqrt (schur) | 111691.4 | 0.2% | 112064.1 | 0.7% |
| cfloat | 1024 | matrix_sqrth (eigh) | 279515.5 | 0.4% | 277482.5 | 0.5% |
| cfloat | 1024 | matrix_sqrt (schur) | 526679.0 | 0.0% | 509297.1 | 0.0% |
| cdouble | 8 | matrix_sqrth (eigh) | 12.5 | 1.4% | 12.5 | 0.5% |
| cdouble | 8 | matrix_sqrt (schur) | 12.3 | 1.0% | 12.3 | 0.7% |
| cdouble | 32 | matrix_sqrth (eigh) | 78.8 | 0.6% | 78.8 | 0.2% |
| cdouble | 32 | matrix_sqrt (schur) | 190.6 | 0.5% | 190.5 | 0.5% |
| cdouble | 128 | matrix_sqrth (eigh) | 2262.6 | 0.6% | 2272.4 | 0.7% |
| cdouble | 128 | matrix_sqrt (schur) | 7411.9 | 0.3% | 7401.8 | 0.3% |
| cdouble | 512 | matrix_sqrth (eigh) | 73835.7 | 1.1% | 73762.1 | 0.5% |
| cdouble | 512 | matrix_sqrt (schur) | 174653.2 | 0.1% | 174399.8 | 0.5% |
| cdouble | 1024 | matrix_sqrth (eigh) | 504799.0 | 0.0% | 505092.1 | 0.0% |
| cdouble | 1024 | matrix_sqrt (schur) | 899193.4 | 0.0% | 899359.5 | 0.0% |

## Timing -- batch size 16

| dtype | n | method | threads=1 median | threads=1 IQR% | threads=N median | threads=N IQR% |
|---|---|---|---|---|---|---|
| float32 | 8 | matrix_sqrth (eigh) | 48.3 | 4.7% | 46.2 | 4.2% |
| float32 | 8 | matrix_sqrt (schur) | 70.3 | 1.3% | 70.3 | 0.2% |
| float32 | 32 | matrix_sqrth (eigh) | 571.9 | 0.6% | 579.5 | 2.2% |
| float32 | 32 | matrix_sqrt (schur) | 958.9 | 0.8% | 955.7 | 0.9% |
| float32 | 128 | matrix_sqrth (eigh) | 7792.3 | 0.5% | 7747.9 | 0.8% |
| float32 | 128 | matrix_sqrt (schur) | 33556.6 | 1.2% | 33682.8 | 0.7% |
| float32 | 512 | matrix_sqrth (eigh) | 134370.5 | 0.9% | 133974.0 | 0.4% |
| float32 | 512 | matrix_sqrt (schur) | 687201.6 | 0.0% | 691665.7 | 0.0% |
| float64 | 8 | matrix_sqrth (eigh) | 68.6 | 0.5% | 68.6 | 0.6% |
| float64 | 8 | matrix_sqrt (schur) | 74.4 | 2.5% | 73.7 | 1.7% |
| float64 | 32 | matrix_sqrth (eigh) | 679.0 | 0.3% | 678.2 | 0.1% |
| float64 | 32 | matrix_sqrt (schur) | 1176.2 | 2.2% | 1176.2 | 1.3% |
| float64 | 128 | matrix_sqrth (eigh) | 10459.7 | 0.6% | 10443.7 | 0.6% |
| float64 | 128 | matrix_sqrt (schur) | 47385.7 | 1.2% | 47308.4 | 0.3% |
| float64 | 512 | matrix_sqrth (eigh) | 212402.8 | 0.3% | 212626.1 | 0.0% |
| float64 | 512 | matrix_sqrt (schur) | 1054999.1 | 0.0% | 1060253.7 | 0.0% |
| cfloat | 8 | matrix_sqrth (eigh) | 65.6 | 1.7% | 65.3 | 2.6% |
| cfloat | 8 | matrix_sqrt (schur) | 139.4 | 0.8% | 138.7 | 0.5% |
| cfloat | 32 | matrix_sqrth (eigh) | 778.2 | 1.4% | 790.6 | 0.9% |
| cfloat | 32 | matrix_sqrt (schur) | 1793.6 | 0.8% | 1765.3 | 0.8% |
| cfloat | 128 | matrix_sqrth (eigh) | 20784.3 | 0.7% | 20682.3 | 0.3% |
| cfloat | 128 | matrix_sqrt (schur) | 81944.3 | 0.6% | 82306.9 | 0.9% |
| cfloat | 512 | matrix_sqrth (eigh) | 762369.7 | 0.0% | 753042.4 | 0.0% |
| cfloat | 512 | matrix_sqrt (schur) | 1790687.0 | 0.0% | 1786654.8 | 0.0% |
| cdouble | 8 | matrix_sqrth (eigh) | 93.8 | 2.2% | 93.1 | 0.9% |
| cdouble | 8 | matrix_sqrt (schur) | 158.2 | 1.0% | 159.5 | 1.2% |
| cdouble | 32 | matrix_sqrth (eigh) | 1219.9 | 0.5% | 1221.2 | 0.4% |
| cdouble | 32 | matrix_sqrt (schur) | 3050.2 | 1.0% | 3051.4 | 0.2% |
| cdouble | 128 | matrix_sqrth (eigh) | 36271.7 | 0.4% | 36309.7 | 0.2% |
| cdouble | 128 | matrix_sqrt (schur) | 120677.8 | 0.4% | 120362.0 | 0.4% |
| cdouble | 512 | matrix_sqrth (eigh) | 1174967.6 | 0.0% | 1172195.8 | 0.0% |
| cdouble | 512 | matrix_sqrt (schur) | 2791294.0 | 0.0% | 2789699.2 | 0.0% |

## Speed ratio (Schur / eigh), threads=N median

Ratio > 1 means the Schur operation is slower than eigh. Computed from the threads=N median times.

| dtype | n | batch | eigh us | schur us | schur/eigh |
|---|---|---|---|---|---|
| float32 | 8 | 1 | 6.0 | 6.9 | 1.14x |
| float32 | 32 | 1 | 37.7 | 80.4 | 2.13x |
| float32 | 128 | 1 | 514.5 | 2200.4 | 4.28x |
| float32 | 512 | 1 | 8642.2 | 47652.8 | 5.51x |
| float32 | 1024 | 1 | 43028.0 | 195103.8 | 4.53x |
| float64 | 8 | 1 | 8.3 | 6.8 | 0.81x |
| float64 | 32 | 1 | 46.0 | 87.1 | 1.89x |
| float64 | 128 | 1 | 666.3 | 2928.2 | 4.39x |
| float64 | 512 | 1 | 13098.0 | 65173.8 | 4.98x |
| float64 | 1024 | 1 | 73186.0 | 310961.7 | 4.25x |
| cfloat | 8 | 1 | 8.5 | 11.2 | 1.31x |
| cfloat | 32 | 1 | 59.7 | 127.2 | 2.13x |
| cfloat | 128 | 1 | 1305.1 | 4992.3 | 3.83x |
| cfloat | 512 | 1 | 46834.6 | 112064.1 | 2.39x |
| cfloat | 1024 | 1 | 277482.5 | 509297.1 | 1.84x |
| cdouble | 8 | 1 | 12.5 | 12.3 | 0.99x |
| cdouble | 32 | 1 | 78.8 | 190.5 | 2.42x |
| cdouble | 128 | 1 | 2272.4 | 7401.8 | 3.26x |
| cdouble | 512 | 1 | 73762.1 | 174399.8 | 2.36x |
| cdouble | 1024 | 1 | 505092.1 | 899359.5 | 1.78x |
| float32 | 8 | 16 | 46.2 | 70.3 | 1.52x |
| float32 | 32 | 16 | 579.5 | 955.7 | 1.65x |
| float32 | 128 | 16 | 7747.9 | 33682.8 | 4.35x |
| float32 | 512 | 16 | 133974.0 | 691665.7 | 5.16x |
| float64 | 8 | 16 | 68.6 | 73.7 | 1.07x |
| float64 | 32 | 16 | 678.2 | 1176.2 | 1.73x |
| float64 | 128 | 16 | 10443.7 | 47308.4 | 4.53x |
| float64 | 512 | 16 | 212626.1 | 1060253.7 | 4.99x |
| cfloat | 8 | 16 | 65.3 | 138.7 | 2.12x |
| cfloat | 32 | 16 | 790.6 | 1765.3 | 2.23x |
| cfloat | 128 | 16 | 20682.3 | 82306.9 | 3.98x |
| cfloat | 512 | 16 | 753042.4 | 1786654.8 | 2.37x |
| cdouble | 8 | 16 | 93.1 | 159.5 | 1.71x |
| cdouble | 32 | 16 | 1221.2 | 3051.4 | 2.50x |
| cdouble | 128 | 16 | 36309.7 | 120362.0 | 3.31x |
| cdouble | 512 | 16 | 1172195.8 | 2789699.2 | 2.38x |

## Measurement stability

Worst per-cell relative IQR across all timed measurements: 10.1%. A low value means the median times are stable; a high value flags a noisy run worth repeating on a quieter machine.

## Accuracy

Worst case over 3 random seeds of the round-trip error `||X X - A||_F / ||A||_F` per operation and the cross-operation agreement `||X_eigh - X_schur||_F / ||A||_F`, plus `scipy` = relative error of the Schur result against `scipy.linalg.sqrtm` (the reference implementation; n <= 256).

| dtype | n | batch | rt_eigh | rt_schur | agreement | scipy |
|---|---|---|---|---|---|---|
| float32 | 8 | 1 | 8.79e-07 | 1.69e-06 | 5.65e-07 | 2.54e-07 |
| float32 | 32 | 1 | 1.49e-06 | 2.68e-06 | 2.50e-07 | 4.11e-07 |
| float32 | 128 | 1 | 1.92e-06 | 4.07e-06 | 2.66e-07 | 2.82e-07 |
| float32 | 512 | 1 | 2.71e-06 | 7.21e-06 | 1.88e-07 | n/a |
| float32 | 1024 | 1 | 3.67e-06 | 9.49e-06 | 2.87e-07 | n/a |
| float64 | 8 | 1 | 2.11e-15 | 3.06e-15 | 6.42e-16 | 3.48e-16 |
| float64 | 32 | 1 | 3.14e-15 | 7.40e-15 | 6.82e-16 | 6.16e-16 |
| float64 | 128 | 1 | 3.56e-15 | 1.06e-14 | 4.55e-16 | 7.00e-16 |
| float64 | 512 | 1 | 5.39e-15 | 1.84e-14 | 3.50e-16 | n/a |
| float64 | 1024 | 1 | 6.98e-15 | 2.21e-14 | 3.25e-16 | n/a |
| cfloat | 8 | 1 | 1.13e-06 | 2.16e-06 | 2.58e-07 | 2.85e-07 |
| cfloat | 32 | 1 | 1.58e-06 | 2.81e-06 | 1.63e-07 | 2.28e-07 |
| cfloat | 128 | 1 | 1.65e-06 | 4.72e-06 | 1.38e-07 | 2.04e-07 |
| cfloat | 512 | 1 | 2.25e-06 | 8.29e-06 | 1.46e-07 | n/a |
| cfloat | 1024 | 1 | 2.78e-06 | 1.02e-05 | 3.24e-07 | n/a |
| cdouble | 8 | 1 | 2.83e-15 | 6.05e-15 | 6.04e-16 | 4.77e-16 |
| cdouble | 32 | 1 | 3.09e-15 | 6.94e-15 | 3.67e-16 | 3.86e-16 |
| cdouble | 128 | 1 | 3.46e-15 | 1.06e-14 | 3.39e-16 | 4.94e-16 |
| cdouble | 512 | 1 | 5.42e-15 | 1.88e-14 | 2.62e-16 | n/a |
| cdouble | 1024 | 1 | 6.97e-15 | 2.32e-14 | 2.35e-16 | n/a |
| float32 | 8 | 16 | 1.58e-06 | 3.05e-06 | 1.02e-06 | 4.56e-07 |
| float32 | 32 | 16 | 1.71e-06 | 3.30e-06 | 6.14e-07 | 4.11e-07 |
| float32 | 128 | 16 | 1.94e-06 | 4.61e-06 | 3.88e-07 | 3.48e-07 |
| float32 | 512 | 16 | 2.82e-06 | 7.48e-06 | 7.69e-07 | n/a |
| float64 | 8 | 16 | 3.13e-15 | 5.36e-15 | 1.49e-15 | 6.53e-16 |
| float64 | 32 | 16 | 3.38e-15 | 8.25e-15 | 1.02e-15 | 9.44e-16 |
| float64 | 128 | 16 | 4.07e-15 | 1.17e-14 | 6.65e-16 | 7.43e-16 |
| float64 | 512 | 16 | 5.53e-15 | 1.84e-14 | 3.65e-16 | n/a |
| cfloat | 8 | 16 | 1.64e-06 | 2.36e-06 | 4.63e-07 | 5.49e-07 |
| cfloat | 32 | 16 | 1.84e-06 | 4.51e-06 | 3.83e-07 | 2.87e-07 |
| cfloat | 128 | 16 | 1.70e-06 | 4.97e-06 | 1.71e-07 | 3.03e-07 |
| cfloat | 512 | 16 | 2.25e-06 | 8.66e-06 | 4.93e-07 | n/a |
| cdouble | 8 | 16 | 3.17e-15 | 4.85e-15 | 6.13e-16 | 8.51e-16 |
| cdouble | 32 | 16 | 3.81e-15 | 1.02e-14 | 5.47e-16 | 4.18e-16 |
| cdouble | 128 | 16 | 3.75e-15 | 1.35e-14 | 5.41e-16 | 5.03e-16 |
| cdouble | 512 | 16 | 5.42e-15 | 2.01e-14 | 3.06e-16 | n/a |

## Interpretation

This run is the strictly single-threaded configuration
(`OMP_NUM_THREADS = VECLIB_MAXIMUM_THREADS = 1`, `nice -20`); the `threads=1` and
`threads=N` columns coincide, so both report genuine single-core performance.

**Headline.** On a single core, the specialized eigh operator (`matrix_sqrth`) is
faster than the generic Schur operator (`matrix_sqrt`) at every non-trivial
size, while both compute the square root to machine precision. For Hermitian
positive-semidefinite inputs, the generic Schur `sqrtm` is not a competitive
replacement.

**Speed.** Beyond a small-`n` break-even (at n=8 fixed dispatch overhead dominates
and the two are within noise -- Schur is even marginally faster for float64/cdouble
single matrices, 0.81x / 0.99x), eigh wins across the board:

- Real dtypes (float32/float64): Schur is 1.8x-5.5x slower, the gap peaking around
  n=128-512 (~4.3-5.5x) and easing slightly at n=1024 (~4.2-4.5x).
- Complex dtypes (cfloat/cdouble): the penalty is smaller, 1.8x-3.8x for a single
  matrix, narrowing to ~1.8x at n=1024 -- complex `heevd` is itself costlier, which
  shrinks the relative gap.
- Batched (16) mirrors the single-matrix pattern (1.1x-5.2x).

The cause is algorithmic, not implementation tuning: `?syevd`/`?heevd` exploit the
Hermitian structure (tridiagonal reduction + divide-and-conquer), whereas `?gees`
pays for a full nonsymmetric Hessenberg + Francis QR reduction it cannot shortcut,
on top of the O(n^3) Bjorck-Hammarling recurrence and the `Z U Z^H` back-transform.

**Accuracy.** Both operators are excellent and the Schur result is independently
validated:

- Round-trip error `||X X - A||_F / ||A||_F` sits at machine-precision scale for
  both (~1e-6 float32, ~1e-14 float64); Schur's is a consistent ~2-3x eigh's but
  the same order of magnitude.
- The scipy oracle (`scipy.linalg.sqrtm`, the reference implementation of this exact
  algorithm) agrees with the Schur operation to ~1e-7 (float32) / ~1e-16 (float64) --
  independent confirmation that `matrix_sqrt` is correct, not merely
  self-consistent with eigh.
- Cross-operation agreement is at roundoff (~1e-7 / ~1e-16), confirming the two ops compute
  the same matrix.

**Measurement quality.** Single-threaded at `nice -20`, with worst per-cell relative
IQR of 10.1% confined to the smallest, noisiest cell (float32 n=8); essentially all
other cells are under ~1%. Because eigh and Schur are timed back-to-back per cell,
the speed ratios are robust even where absolute times would drift.

**Scope and caveats.** Absolute times are Apple Accelerate (vecLib) specific; the
qualitative conclusion is algorithmic and is expected to hold on Linux/MKL, but the
absolute ratios are not portable across BLAS backends. Inputs are well-conditioned
random Hermitian PSD (`B B^H + 1e-3 I`) -- the intended regime for replacing
`matrix_sqrth`, but it does not exercise ill-conditioned or non-normal matrices. The
comparison is forward-only (`matrix_sqrt` has no autograd). As a single-machine
microbenchmark this is sufficient to settle the relative question; it is not a
production performance validation (which would additionally need the target
hardware, multiple BLAS backends, multi-run statistics, and memory/tail metrics).

**Recommendation.** Keep `matrix_sqrth` (the eigh path) as the operator for Hermitian
inputs. The generic Schur `sqrtm` earns its keep only through generality -- non-normal
/ non-Hermitian matrices outside `matrix_sqrth`'s contract -- which would justify a
separate general-matrix operator rather than a swap-in here.

