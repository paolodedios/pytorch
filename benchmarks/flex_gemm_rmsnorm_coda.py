from __future__ import annotations

import argparse
from dataclasses import dataclass
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
        kernel_options={"backend": config.backend},
    )
    inv_rms = torch.rsqrt(partial_sums.sum(-1, keepdim=True) / hidden + config.eps)

    def second_epilogue(acc: torch.Tensor) -> torch.Tensor:
        return (acc.float() * inv_rms).to(acc.dtype)

    return flex_gemm(
        torch.ops.aten.mm.default,
        (h_gamma, w1),
        second_epilogue,
        kernel_options={"backend": config.backend},
    )


_COMPILED_FORWARD_CACHE: dict[CodaRmsNormConfig, Any] = {}
_COMPILED_BACKWARD_CACHE: dict[CodaRmsNormConfig, Any] = {}


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
        kernel_options={"backend": config.backend},
    )
    inv_rms = torch.rsqrt(partial_sums.sum(-1, keepdim=True) / hidden + config.eps)
    s = (grad_out.float() * out.float()).sum(-1, keepdim=True) / hidden

    def grad_h_epilogue(acc: torch.Tensor) -> torch.Tensor:
        grad_h2 = acc.float()
        h_f = h.float()
        inv = inv_rms
        grad_normed = grad_h2 * gamma_row.float()
        return (inv * (grad_normed - h_f * inv * s)).to(acc.dtype)

    grad_h = flex_gemm(
        torch.ops.aten.mm.default,
        (grad_out, w1.mT),
        grad_h_epilogue,
        kernel_options={"backend": config.backend},
    )

    def grad_x_epilogue(acc: torch.Tensor) -> torch.Tensor:
        return acc.to(x.dtype)

    grad_x = flex_gemm(
        torch.ops.aten.mm.default,
        (grad_h, w0.mT),
        grad_x_epilogue,
        kernel_options={"backend": config.backend},
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
    ) -> torch.Tensor:
        config = CodaRmsNormConfig(eps=eps, partial_group=partial_group)
        with torch.no_grad():
            out = compiled_coda_rmsnorm_gemm_forward(
                x, w0, residual, gamma, w1, config
            )
        ctx.save_for_backward(x, w0, residual, gamma, w1, out)
        ctx.eps = eps
        ctx.partial_group = partial_group
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        x, w0, residual, gamma, w1, out = ctx.saved_tensors
        config = CodaRmsNormConfig(eps=ctx.eps, partial_group=ctx.partial_group)
        with torch.no_grad():
            grad_x, grad_w0, grad_residual, grad_gamma, grad_w1 = (
                compiled_coda_rmsnorm_gemm_backward_flex(
                    x, w0, residual, gamma, w1, out, grad_out, config
                )
            )
        return grad_x, grad_w0, grad_residual, grad_gamma, grad_w1, None, None


def coda_rmsnorm_gemm(
    x: torch.Tensor,
    w0: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    w1: torch.Tensor,
    eps: float = 1e-5,
    partial_group: int = 32,
) -> torch.Tensor:
    """Apply the custom autograd CODA RMSNorm-GEMM prototype."""
    return CodaRmsNormGemmFunction.apply(x, w0, residual, gamma, w1, eps, partial_group)


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
        torch.randn(hidden, device="cuda", dtype=dtype) * 0.25,
        torch.randn(hidden, n, device="cuda", dtype=dtype) * 0.25,
    )


def assert_forward_codegen(code: str) -> None:
    """Check that the generated forward keeps both GEMMs on the QUACK path."""
    required = (
        "gemm_epilogue(",
        "local_reduce_source_from_epilogue=True",
        "epilogue_args=",
    )
    missing = [item for item in required if item not in code]
    if missing:
        raise AssertionError(f"generated code is missing {missing}")
    if "extern_kernels.mm" in code:
        raise AssertionError("generated code fell back to extern_kernels.mm")
    if code.count("gemm_epilogue(") < 2:
        raise AssertionError("expected two fused GEMM epilogue calls")


def check_forward(args: argparse.Namespace) -> None:
    """Compile and verify the fused CODA forward expression."""
    dtype = getattr(torch, args.dtype)
    config = CodaRmsNormConfig(eps=args.eps, partial_group=args.partial_group)
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
    if code.count("gemm_epilogue(") < 3:
        raise AssertionError("expected recompute, grad_h, and grad_x FlexGEMM calls")
    if "epilogue_args=" not in code:
        raise AssertionError("expected captured RMSNorm state in backward epilogue")


def check_flex_backward(args: argparse.Namespace) -> None:
    """Compile and verify the FlexGEMM-shaped backward helper directly."""
    dtype = getattr(torch, args.dtype)
    config = CodaRmsNormConfig(eps=args.eps, partial_group=args.partial_group)
    inputs = make_inputs(args.m, args.k, args.hidden, args.n, dtype)
    out = reference_reparameterized_coda(*inputs, args.eps).detach()
    grad = torch.randn_like(out)

    def backward_fn(
        x: torch.Tensor,
        w0: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        w1: torch.Tensor,
        out: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return coda_rmsnorm_gemm_backward_flex(
            x, w0, residual, gamma, w1, out, grad_out, config
        )

    actual_grads, codes = run_and_get_code(
        torch.compile(backward_fn, backend="inductor", fullgraph=True),
        *inputs,
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

    actual = coda_rmsnorm_gemm(*inputs, args.eps, args.partial_group)
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
    parser.add_argument("--skip-autograd", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the forward codegen and custom-autograd smoke checks."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CODA FlexGEMM prototype")
    args = parse_args()
    check_forward(args)
    if not args.skip_autograd:
        check_flex_backward(args)
        check_autograd(args)


if __name__ == "__main__":
    main()
