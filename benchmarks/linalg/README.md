# Linalg benchmarks

Run every benchmark from the repo root as a module with a from-source install.

## matrix_sqrt

`matrix_sqrt_benchmark.py` compares two ATen ops for the principal square root of Hermitian PSD matrices, on speed and accuracy:

- `matrix_sqrth` uses the eigh path (`?syevd`/`?heevd` from LAPACK, symmetry-exploiting).
- `matrix_sqrt` uses generic Schur path (`?gees` + Bjorck-Hammarling, `?trsyl`).

It sweeps size, batch, and dtype, timing each operation with `torch.utils.benchmark.Timer.blocked_autorange` reporting round-trip error, cross-operation agreement, and relative error against the reference `scipy.linalg.sqrtm`. CPU only. `--label NAME` tags the report header with the scenario; `--quick` is a fast sanity sweep.

### Run

This test was ran on a MacBook Pro M4 Pro with 24GB of memory.

```bash
sudo OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n -20 \
  .venv/bin/python -m benchmarks.linalg.matrix_sqrt_benchmark \
  --label 1-core --out benchmarks/linalg/matrix_sqrt_results_1core.md
```

Only the single-core run is reported. On Apple Silicon, Accelerate (vecLib) manages its own LAPACK threading and does not parallelize these factorizations (`?syevd`/`?heevd`, `?gees`/`?trsyl`) at these sizes, so raising the caps to $8$ leaves both the LAPACK-dominated times and the `schur`/`eigh` ratio unchanged. A multi-core run adds nothing on this backend.

## matrix_sqrth

`matrix_sqrth_benchmark.py` times the principal square root of Hermitian PSD matrices via two PyTorch-native pipelines, both backed by ATen's LAPACK eigh (`dsyevd`/`zheevd`).
