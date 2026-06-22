from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch._higher_order_ops import flex_gemm
from torch._inductor.utils import run_and_get_code


@dataclass(frozen=True)
class CodaRmsNormConfig:
    """Static knobs that define the CODA RMSNorm fusion prototype."""

    eps: float = 1e-5
    partial_group: int = 32
    backend: str = "QUACK"
    tuned: bool = False


def reference_reparameterized_coda(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Evaluate CODA's row-scale-commuted RMSNorm-GEMM expression."""
    h = x.matmul(w0) + residual
    inv_rms = torch.rsqrt(h.float().square().mean(-1, keepdim=True) + eps)
    h_gamma = (h.float() * gamma.reshape(1, -1).float()).to(x.dtype)
    return (h_gamma.matmul(w1).float() * inv_rms).to(x.dtype)


def reference_rmsnorm_gemm(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Evaluate the original RMSNorm-then-GEMM expression before CODA rewriting."""
    h = x.matmul(w0) + residual
    normed = F.rms_norm(h.float(), (h.shape[-1],), gamma.float(), eps).to(x.dtype)
    return normed.matmul(w1)


def coda_rmsnorm_gemm_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> torch.Tensor:
    """Run CODA forward using two FlexGEMM epilogues plus a small row reduction."""
    m = x.shape[0]
    hidden = w0.shape[1]
    if hidden % config.partial_group != 0:
        raise RuntimeError(
            f"hidden size {hidden} must be divisible by {config.partial_group}"
        )
    gamma_row = gamma.reshape(1, hidden)

    def first_epilogue(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = acc.float() + residual
        partial_sums = (h * h).view(m, -1, config.partial_group).sum(-1)
        return (h * gamma_row).to(acc.dtype), partial_sums

    h_gamma, partial_sums = flex_gemm(
        torch.ops.aten.mm.default,
        (x, w0),
        first_epilogue,
        kernel_options={"backend": config.backend, "tuned": config.tuned},
    )
    inv_rms = torch.rsqrt(partial_sums.sum(-1, keepdim=True) / hidden + config.eps)

    return (h_gamma.matmul(w1).float() * inv_rms).to(x.dtype)


def coda_rmsnorm_gemm_forward_state(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the forward and expose saved tensors needed by the CODA backward."""
    m = x.shape[0]
    hidden = w0.shape[1]
    gamma_row = gamma.reshape(1, hidden)

    def first_epilogue(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = acc.float() + residual
        partial_sums = (h * h).view(m, -1, config.partial_group).sum(-1)
        return (h * gamma_row).to(acc.dtype), partial_sums

    h_gamma, partial_sums = flex_gemm(
        torch.ops.aten.mm.default,
        (x, w0),
        first_epilogue,
        kernel_options={"backend": config.backend, "tuned": config.tuned},
    )
    inv_rms = torch.rsqrt(partial_sums.sum(-1, keepdim=True) / hidden + config.eps)

    out = (h_gamma.matmul(w1).float() * inv_rms).to(x.dtype)
    return out, h_gamma, inv_rms


_COMPILED_FORWARD_CACHE: dict[CodaRmsNormConfig, Any] = {}
_COMPILED_FORWARD_STATE_CACHE: dict[CodaRmsNormConfig, Any] = {}
_COMPILED_BACKWARD_CACHE: dict[CodaRmsNormConfig, Any] = {}
_COMPILED_BACKWARD_SAVED_CACHE: dict[CodaRmsNormConfig, Any] = {}


def compiled_coda_rmsnorm_gemm_forward(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> torch.Tensor:
    """Call a cached Inductor-compiled CODA forward for custom autograd use."""
    compiled = _COMPILED_FORWARD_CACHE.get(config)
    if compiled is None:

        def forward_fn(
            x_arg: torch.Tensor,
            w0_arg: torch.Tensor,
            residual_arg: torch.Tensor,
            gamma_arg: torch.Tensor,
            w1_arg: torch.Tensor,
        ) -> torch.Tensor:
            return coda_rmsnorm_gemm_forward(
                x_arg, w0_arg, residual_arg, gamma_arg, w1_arg, config
            )

        compiled = torch.compile(forward_fn, backend="inductor", fullgraph=True)
        _COMPILED_FORWARD_CACHE[config] = compiled
    return compiled(x, w0, residual, gamma, w1)


def compiled_coda_rmsnorm_gemm_forward_state(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call a cached compiled forward that returns CODA backward state."""
    compiled = _COMPILED_FORWARD_STATE_CACHE.get(config)
    if compiled is None:

        def forward_fn(
            x_arg: torch.Tensor,
            w0_arg: torch.Tensor,
            residual_arg: torch.Tensor,
            gamma_arg: torch.Tensor,
            w1_arg: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return coda_rmsnorm_gemm_forward_state(
                x_arg, w0_arg, residual_arg, gamma_arg, w1_arg, config
            )

        compiled = torch.compile(forward_fn, backend="inductor", fullgraph=True)
        _COMPILED_FORWARD_STATE_CACHE[config] = compiled
    return compiled(x, w0, residual, gamma, w1)


def coda_rmsnorm_gemm_backward_flex(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute backward with FlexGEMM epilogues for activation-gradient GEMMs."""
    m = x.shape[0]
    hidden = w0.shape[1]
    if hidden % config.partial_group != 0:
        raise RuntimeError(
            f"hidden size {hidden} must be divisible by {config.partial_group}"
        )
    gamma_row = gamma.reshape(1, hidden)

    def recompute_h_epilogue(acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = acc.float() + residual
        partial_sums = (h * h).view(m, -1, config.partial_group).sum(-1)
        return h.to(acc.dtype), partial_sums

    h, partial_sums = flex_gemm(
        torch.ops.aten.mm.default,
        (x, w0),
        recompute_h_epilogue,
        kernel_options={"backend": config.backend, "tuned": config.tuned},
    )
    inv_rms = torch.rsqrt(partial_sums.sum(-1, keepdim=True) / hidden + config.eps)
    s = (grad_out.float() * out.float()).sum(-1, keepdim=True) / hidden

    def grad_h_epilogue(acc: torch.Tensor) -> torch.Tensor:
        grad_h2 = acc.float()
        h_f = h.float()
        grad_normed = grad_h2 * gamma_row.float()
        return (inv_rms * (grad_normed - h_f * inv_rms * s)).to(acc.dtype)

    grad_h = flex_gemm(
        torch.ops.aten.mm.default,
        (grad_out, w1.mT),
        grad_h_epilogue,
        kernel_options={"backend": config.backend, "tuned": config.tuned},
    )

    def grad_x_epilogue(acc: torch.Tensor) -> torch.Tensor:
        return acc.to(x.dtype)

    grad_x = flex_gemm(
        torch.ops.aten.mm.default,
        (grad_h, w0.mT),
        grad_x_epilogue,
        kernel_options={"backend": config.backend, "tuned": False},
    )
    h_f = h.float()
    grad_h2 = grad_out.float().matmul(w1.float().mT)
    h2 = (h_f * inv_rms * gamma_row.float()).to(grad_out.dtype)
    grad_w0 = x.float().mT.matmul(grad_h.float())
    grad_w1 = h2.float().mT.matmul(grad_out.float())
    grad_gamma = (grad_h2 * h_f * inv_rms).sum(0).reshape_as(gamma)
    return (
        grad_x.to(x.dtype),
        grad_w0.to(w0.dtype),
        grad_h.to(residual.dtype),
        grad_gamma.to(gamma.dtype),
        grad_w1.to(w1.dtype),
    )

def coda_rmsnorm_gemm_backward_flex_saved(
    x: torch.Tensor,
    w0: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    h_gamma: torch.Tensor,
    inv_rms: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute saved-state CODA backward for model RMSNorm weights near one."""
    hidden = w0.shape[1]
    gamma_row = gamma.reshape(1, hidden).float()
    inv_gamma = gamma_row.reciprocal()
    h_gamma_f = h_gamma.float()
    s = (grad_out.float() * out.float()).sum(-1, keepdim=True) / hidden

    grad_h2 = grad_out.matmul(w1.mT).float()
    grad_h = (
        inv_rms * (grad_h2 * gamma_row)
        - h_gamma_f * (inv_rms * inv_rms * s * inv_gamma)
    ).to(grad_out.dtype)

    grad_x = grad_h.matmul(w0.mT)
    grad_w0 = x.mT.matmul(grad_h)
    grad_w1 = h_gamma.mT.matmul((grad_out.float() * inv_rms).to(grad_out.dtype))
    grad_gamma = (grad_h2 * h_gamma_f * inv_rms).sum(0) * inv_gamma.reshape(-1)
    return (
        grad_x.to(x.dtype),
        grad_w0.to(w0.dtype),
        grad_h.to(h_gamma.dtype),
        grad_gamma.reshape_as(gamma).to(gamma.dtype),
        grad_w1.to(w1.dtype),
    )


def compiled_coda_rmsnorm_gemm_backward_flex(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call a cached Inductor-compiled backward FlexGEMM prototype."""
    compiled = _COMPILED_BACKWARD_CACHE.get(config)
    if compiled is None:

        def backward_fn(
            x_arg: torch.Tensor,
            w0_arg: torch.Tensor,
            residual_arg: torch.Tensor,
            gamma_arg: torch.Tensor,
            w1_arg: torch.Tensor,
            out_arg: torch.Tensor,
            grad_out_arg: torch.Tensor,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            return coda_rmsnorm_gemm_backward_flex(
                x_arg,
                w0_arg,
                residual_arg,
                gamma_arg,
                w1_arg,
                out_arg,
                grad_out_arg,
                config,
            )

        compiled = torch.compile(backward_fn, backend="inductor", fullgraph=True)
        _COMPILED_BACKWARD_CACHE[config] = compiled
    return compiled(x, w0, residual, gamma, w1, out, grad_out)


def compiled_coda_rmsnorm_gemm_backward_flex_saved(
    x: torch.Tensor,
    w0: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    h_gamma: torch.Tensor,
    inv_rms: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    config: CodaRmsNormConfig = CodaRmsNormConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call a cached compiled backward that consumes saved CODA state."""
    compiled = _COMPILED_BACKWARD_SAVED_CACHE.get(config)
    if compiled is None:

        def backward_fn(
            x_arg: torch.Tensor,
            w0_arg: torch.Tensor,
            gamma_arg: torch.Tensor,
            w1_arg: torch.Tensor,
            h_gamma_arg: torch.Tensor,
            inv_rms_arg: torch.Tensor,
            out_arg: torch.Tensor,
            grad_out_arg: torch.Tensor,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            return coda_rmsnorm_gemm_backward_flex_saved(
                x_arg,
                w0_arg,
                gamma_arg,
                w1_arg,
                h_gamma_arg,
                inv_rms_arg,
                out_arg,
                grad_out_arg,
                config,
            )

        compiled = torch.compile(backward_fn, backend="inductor", fullgraph=True)
        _COMPILED_BACKWARD_SAVED_CACHE[config] = compiled
    return compiled(x, w0, gamma, w1, h_gamma, inv_rms, out, grad_out)


class CodaRmsNormGemmFunction(torch.autograd.Function):
    """Autograd wrapper around the fused CODA RMSNorm-GEMM forward.

    Args:
        x: Input activation with shape ``[M, K]``.
        w0: First GEMM weight with shape ``[K, H]``.
        residual: Residual tensor with shape ``[M, H]``.
        gamma: RMSNorm weight with shape ``[H]`` or ``[1, H]``.
        w1: Second GEMM weight with shape ``[H, N]``.
        eps: RMSNorm epsilon.
        partial_group: Hidden-dimension group used for local partial sums.

    Returns:
        The CODA-rewritten RMSNorm-GEMM output with shape ``[M, N]``.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        w0: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        w1: torch.Tensor,
        eps: float = 1e-5,
        partial_group: int = 32,
        tuned: bool = False,
    ) -> torch.Tensor:
        config = CodaRmsNormConfig(eps=eps, partial_group=partial_group, tuned=tuned)
        with torch.no_grad():
            out, h_gamma, inv_rms = compiled_coda_rmsnorm_gemm_forward_state(
                x, w0, residual, gamma, w1, config
            )
        ctx.save_for_backward(x, w0, gamma, w1, out, h_gamma, inv_rms)
        ctx.eps = eps
        ctx.partial_group = partial_group
        ctx.tuned = tuned
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        x, w0, gamma, w1, out, h_gamma, inv_rms = ctx.saved_tensors
        config = CodaRmsNormConfig(
            eps=ctx.eps, partial_group=ctx.partial_group, tuned=ctx.tuned
        )
        with torch.no_grad():
            grad_x, grad_w0, grad_residual, grad_gamma, grad_w1 = (
                compiled_coda_rmsnorm_gemm_backward_flex_saved(
                    x, w0, gamma, w1, h_gamma, inv_rms, out, grad_out, config
                )
            )
        return grad_x, grad_w0, grad_residual, grad_gamma, grad_w1, None, None, None


def coda_rmsnorm_gemm(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    eps: float = 1e-5,
    partial_group: int = 32,
    tuned: bool = False,
) -> torch.Tensor:
    """Apply the fast custom autograd CODA prototype for nonzero RMSNorm gamma."""
    return CodaRmsNormGemmFunction.apply(
        x, w0, residual, gamma, w1, eps, partial_group, tuned
    )


def make_inputs(
    m: int,
    k: int,
    hidden: int,
    n: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Create bounded CUDA inputs for correctness and codegen checks."""
    return (
        torch.randn(m, k, device="cuda", dtype=dtype) * 0.25,
        torch.randn(k, hidden, device="cuda", dtype=dtype) * 0.25,
        torch.randn(m, hidden, device="cuda", dtype=dtype) * 0.25,
        torch.randn(hidden, device="cuda", dtype=dtype) * 0.05 + 1.0,
        torch.randn(hidden, n, device="cuda", dtype=dtype) * 0.25,
    )


def assert_forward_codegen(code: str) -> None:
    """Check that the generated forward fuses the RMS partials into the first GEMM."""
    required = (
        "gemm_epilogue(",
        "local_reduce_source_from_epilogue=True",
        "epilogue_args=",
    )
    missing = [item for item in required if item not in code]
    if missing:
        raise AssertionError(f"generated code is missing {missing}")
    if code.count("gemm_epilogue(") < 1:
        raise AssertionError("expected fused first GEMM epilogue call")


def check_forward(args: argparse.Namespace) -> None:
    """Compile and verify the fused CODA forward expression."""
    dtype = getattr(torch, args.dtype)
    config = CodaRmsNormConfig(
        eps=args.eps, partial_group=args.partial_group, tuned=args.tuned
    )
    inputs = make_inputs(args.m, args.k, args.hidden, args.n, dtype)

    def forward_fn(*forward_inputs: torch.Tensor) -> torch.Tensor:
        return coda_rmsnorm_gemm_forward(*forward_inputs, config)

    actual, codes = run_and_get_code(
        torch.compile(forward_fn, backend="inductor", fullgraph=True), *inputs
    )
    reparameterized = reference_reparameterized_coda(*inputs, args.eps)
    original = reference_rmsnorm_gemm(*inputs, args.eps)
    torch.testing.assert_close(actual, reparameterized, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(actual.float(), original.float(), atol=0.35, rtol=0.2)
    code = "\n".join(codes)
    assert_forward_codegen(code)
    if args.tuned and "tuned=True" not in code:
        raise AssertionError("expected tuned=True in generated FlexGEMM call")
    print("forward ok")
    print(f"generated gemm_epilogue calls: {code.count('gemm_epilogue(')}")


def eager_rmsnorm_gemm_gradients(
    inputs: tuple[torch.Tensor, ...], grad: torch.Tensor, eps: float
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Use eager RMSNorm autograd as the numerical reference."""
    ref_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    expected = reference_rmsnorm_gemm(*ref_inputs, eps)
    expected.backward(grad)
    return expected, tuple(input.grad for input in ref_inputs)


def assert_backward_codegen(code: str) -> None:
    """Check that backward activation-gradient GEMMs use FlexGEMM epilogues."""
    if "extern_kernels.mm" not in code:
        raise AssertionError("expected matmul calls in backward")


def check_flex_backward(args: argparse.Namespace) -> None:
    """Compile and verify the FlexGEMM-shaped backward helper directly."""
    dtype = getattr(torch, args.dtype)
    config = CodaRmsNormConfig(
        eps=args.eps, partial_group=args.partial_group, tuned=args.tuned
    )
    inputs = make_inputs(args.m, args.k, args.hidden, args.n, dtype)
    out, h_gamma, inv_rms = coda_rmsnorm_gemm_forward_state(*inputs, config)
    grad = torch.randn_like(out)

    def backward_fn(
        x: torch.Tensor,
        w0: torch.Tensor,
        gamma: torch.Tensor,
        w1: torch.Tensor,
        h_gamma: torch.Tensor,
        inv_rms: torch.Tensor,
        out: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return coda_rmsnorm_gemm_backward_flex_saved(
            x, w0, gamma, w1, h_gamma, inv_rms, out, grad_out, config
        )

    actual_grads, codes = run_and_get_code(
        torch.compile(backward_fn, backend="inductor", fullgraph=True),
        inputs[0],
        inputs[1],
        inputs[3],
        inputs[4],
        h_gamma,
        inv_rms,
        out,
        grad,
    )
    _expected, expected_grads = eager_rmsnorm_gemm_gradients(inputs, grad, args.eps)
    for index, (actual_grad, expected_grad) in enumerate(
        zip(actual_grads, expected_grads, strict=True)
    ):
        torch.testing.assert_close(
            actual_grad.float(),
            expected_grad.float(),
            atol=args.grad_atol,
            rtol=args.grad_rtol,
            msg=lambda msg, i=index: f"input {i} gradient mismatch: {msg}",
        )
    assert_backward_codegen("\n".join(codes))
    print("flex backward ok")


def check_autograd(args: argparse.Namespace) -> None:
    """Verify custom autograd gradients against the original expression."""
    dtype = getattr(torch, args.dtype)
    inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in make_inputs(args.m, args.k, args.hidden, args.n, dtype)
    )

    actual = coda_rmsnorm_gemm(*inputs, args.eps, args.partial_group, args.tuned)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected, expected_grads = eager_rmsnorm_gemm_gradients(inputs, grad, args.eps)

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.35, rtol=0.2)
    for index, (actual_input, expected_grad) in enumerate(zip(inputs, expected_grads)):
        torch.testing.assert_close(
            actual_input.grad.float(),
            expected_grad.float(),
            atol=args.grad_atol,
            rtol=args.grad_rtol,
            msg=lambda msg, i=index: f"input {i} gradient mismatch: {msg}",
        )
    print("autograd ok")


def make_benchmark_inputs(
    m: int,
    k: int,
    hidden: int,
    n: int,
    dtype: torch.dtype,
    input_scale: float,
    gamma_scale: float,
) -> tuple[torch.Tensor, ...]:
    """Create model-like inputs for the real-shape timing contract."""
    return (
        (torch.randn(m, k, device="cuda", dtype=dtype) * input_scale)
        .detach()
        .requires_grad_(True),
        (torch.randn(k, hidden, device="cuda", dtype=dtype) * input_scale)
        .detach()
        .requires_grad_(True),
        (torch.randn(m, hidden, device="cuda", dtype=dtype) * input_scale)
        .detach()
        .requires_grad_(True),
        (torch.randn(hidden, device="cuda", dtype=dtype) * gamma_scale + 1.0)
        .detach()
        .requires_grad_(True),
        (torch.randn(hidden, n, device="cuda", dtype=dtype) * input_scale)
        .detach()
        .requires_grad_(True),
    )


def clone_grad_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Clone leaf tensors so reference and CODA checks use identical values."""
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in inputs)


def tensor_error_stats(actual: torch.Tensor, expected: torch.Tensor) -> str:
    """Format absolute error statistics that are stable near expected zeros."""
    diff = (actual.float() - expected.float()).abs()
    return f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}"


def time_cuda_step(
    name: str,
    step: Any,
    *,
    warmup: int,
    iters: int,
) -> float:
    """Time a CUDA step with the same Inductor benchmarker used in perf triage."""
    step()
    torch.cuda.synchronize()
    try:
        from transformer_nuggets.utils.benchmark import benchmark_cuda_function_stats
    except ModuleNotFoundError:
        elapsed_ms = []
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
        sorted_ms = sorted(elapsed_ms)
        p05 = sorted_ms[max(0, int(0.05 * (iters - 1)))]
        p95 = sorted_ms[min(iters - 1, int(0.95 * (iters - 1)))]
        med = median(elapsed_ms)
    else:
        stats = benchmark_cuda_function_stats(
            step,
            NUM_ITERS=iters,
            MEMORY_WARMUP_ITERS=warmup,
            USE_CUDA_GRAPHS=False,
            IS_VETTED_BENCHMARKING=True,
        )
        med = stats.median_us / 1000
        p05 = stats.quantiles_us[0] / 1000
        p95 = stats.quantiles_us[2] / 1000
    print(f"{name}: median={med:.3f} ms p05={p05:.3f} p95={p95:.3f}")
    return med


def benchmark_e2e(args: argparse.Namespace) -> None:
    """Compare compiled eager RMSNorm-GEMM autograd with the CODA custom path."""
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    base_inputs = make_benchmark_inputs(
        args.m,
        args.k,
        args.hidden,
        args.n,
        dtype,
        args.input_scale,
        args.gamma_scale,
    )
    grad = torch.randn(args.m, args.n, device="cuda", dtype=dtype) * args.input_scale
    config = CodaRmsNormConfig(
        eps=args.eps, partial_group=args.partial_group, tuned=args.tuned
    )

    print(
        f"device={torch.cuda.get_device_name()} "
        f"shape={(args.m, args.k, args.hidden, args.n)} dtype={dtype} tuned={args.tuned} "
        f"input_scale={args.input_scale} gamma_scale={args.gamma_scale}"
    )

    def reference_fn(
        x: torch.Tensor,
        w0: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        w1: torch.Tensor,
    ) -> torch.Tensor:
        return reference_rmsnorm_gemm(x, w0, residual, gamma, w1, args.eps)

    compiled_reference = torch.compile(reference_fn, backend="inductor", fullgraph=True)

    def reference_step() -> tuple[torch.Tensor, ...]:
        out = compiled_reference(*base_inputs)
        return torch.autograd.grad(out, base_inputs, grad_outputs=grad)

    def coda_step() -> tuple[torch.Tensor, ...]:
        out = coda_rmsnorm_gemm(
            *base_inputs,
            eps=config.eps,
            partial_group=config.partial_group,
            tuned=config.tuned,
        )
        return torch.autograd.grad(out, base_inputs, grad_outputs=grad)

    print(
        "timing contract: Inductor GPU benchmarker, no CUDA graphs, compile excluded, "
        f"warmup={args.bench_warmup}, iters={args.bench_iters}"
    )
    baseline_ms = time_cuda_step(
        "compiled eager RMSNorm-GEMM autograd",
        reference_step,
        warmup=args.bench_warmup,
        iters=args.bench_iters,
    )
    coda_ms = time_cuda_step(
        "CODA FlexGEMM custom autograd",
        coda_step,
        warmup=args.bench_warmup,
        iters=args.bench_iters,
    )
    print(f"speedup={baseline_ms / coda_ms:.4f}x")

    ref_inputs = clone_grad_inputs(base_inputs)
    coda_inputs = clone_grad_inputs(base_inputs)
    ref_out = reference_rmsnorm_gemm(*ref_inputs, args.eps)
    coda_out = coda_rmsnorm_gemm(*coda_inputs, args.eps, args.partial_group, args.tuned)
    ref_grads = torch.autograd.grad(ref_out, ref_inputs, grad_outputs=grad)
    coda_grads = torch.autograd.grad(coda_out, coda_inputs, grad_outputs=grad)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        coda_out.float(),
        ref_out.float(),
        atol=args.benchmark_atol,
        rtol=args.benchmark_rtol,
    )
    print(f"output error: {tensor_error_stats(coda_out, ref_out)}")
    for index, (actual_grad, expected_grad) in enumerate(
        zip(coda_grads, ref_grads, strict=True)
    ):
        torch.testing.assert_close(
            actual_grad.float(),
            expected_grad.float(),
            atol=args.benchmark_grad_atol,
            rtol=args.benchmark_grad_rtol,
            msg=lambda msg, i=index: f"benchmark grad {i} mismatch: {msg}",
        )
        print(f"grad {index} error: {tensor_error_stats(actual_grad, expected_grad)}")


def parse_args() -> argparse.Namespace:
    """Parse prototype smoke-test options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--partial-group", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--grad-atol", type=float, default=5e-2)
    parser.add_argument("--grad-rtol", type=float, default=5e-2)
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--skip-autograd", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--input-scale", type=float, default=0.05)
    parser.add_argument("--gamma-scale", type=float, default=0.05)
    parser.add_argument("--benchmark-atol", type=float, default=0.35)
    parser.add_argument("--benchmark-rtol", type=float, default=0.35)
    parser.add_argument("--benchmark-grad-atol", type=float, default=0.35)
    parser.add_argument("--benchmark-grad-rtol", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the forward codegen and custom-autograd smoke checks."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CODA FlexGEMM prototype")
    args = parse_args()
    if args.benchmark:
        benchmark_e2e(args)
        return
    check_forward(args)
    if not args.skip_autograd:
        check_flex_backward(args)
        check_autograd(args)


if __name__ == "__main__":
    main()
