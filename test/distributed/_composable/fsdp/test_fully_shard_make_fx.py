# Owner(s): ["oncall: distributed"]
"""
Standalone test for make_fx tracing of FSDP2-wrapped models.

Uses a local clone of sixlib's minimal_tracer (trace_module) which lifts
params as graph inputs via stateless._reparametrize_module.

Usage:
    torchrun --nproc_per_node=2 test_fully_shard_make_fx.py
"""

import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from minimal_tracer import _collect_fsdp_params_and_metas, run_traced_module, trace_module
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


# =============================================
# Model definitions
# =============================================


class SimpleMLP(nn.Module):
    def __init__(self, dim: int = 32, device: torch.device | None = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4, device=device, bias=False)
        self.fc2 = nn.Linear(dim * 4, dim, device=device, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StackedMLP(nn.Module):
    def __init__(
        self, dim: int = 32, n_layers: int = 3, device: torch.device | None = None
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [SimpleMLP(dim, device=device) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FwdBwdModule(nn.Module):
    """Module that calls autograd.grad in forward, capturing both fwd+bwd."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, fsdp_sharded_params: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        out = self.model(x)
        loss = out.sum()
        if fsdp_sharded_params is not None:
            grads = torch.autograd.grad(
                loss, fsdp_sharded_params, allow_unused=True
            )
            grads = tuple(
                g if g is not None else torch.zeros_like(s)
                for g, s in zip(grads, fsdp_sharded_params)
            )
            return (loss, *grads)
        return loss


# =============================================
# Utilities
# =============================================


def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def build_fsdp_model(dim: int = 32, n_layers: int = 3) -> nn.Module:
    device = torch.device("cuda")
    model = StackedMLP(dim=dim, n_layers=n_layers, device=device)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def test_trace_module():
    """Trace FSDP2 model with forward + backward (fwd+bwd graph)."""
    rank = dist.get_rank()
    model = build_fsdp_model()

    # Trigger lazy init
    x_init = torch.randn(4, 32, device="cuda")
    out_init = model(x_init)
    out_init.sum().backward()
    model.zero_grad()
    if rank == 0:
        log.info("[trace_module] Lazy init done")

    # Wrap model so forward calls autograd.grad (captures fwd+bwd)
    fwd_bwd = FwdBwdModule(model)

    x = torch.randn(4, 32, device="cuda")

    if rank == 0:
        log.info("[trace_module] Starting trace...")

    try:
        traced = trace_module(fwd_bwd, (x,))
    except Exception as e:
        if rank == 0:
            log.info(
                f"[trace_module] Tracing failed: {type(e).__name__}: {e}"
            )
            import traceback
            traceback.print_exc()
        return

    if rank == 0:
        log.info("[trace_module] Tracing complete!")
        log.info("[trace_module] Readable graph:")
        traced.print_readable()

    # Numerical check: compare eager vs traced outputs
    # Note: numerics may not match yet (WIP)

    # Run traced graph
    traced_out = run_traced_module(traced, fwd_bwd, (x,))
    traced_loss = traced_out[0]
    traced_grads = traced_out[1:]

    # Run eager: forward + backward to get sharded param grads
    model.zero_grad()
    eager_loss = model(x).sum()
    eager_loss.backward()
    fsdp_params, _ = _collect_fsdp_params_and_metas(fwd_bwd)
    eager_grads = [fp.sharded_param.grad for fp in fsdp_params]

    if rank == 0:
        # Compare loss
        loss_diff = (eager_loss - traced_loss).abs().item()
        log.info(f"[numerical] loss: max_diff={loss_diff:.6e}")

        # Compare gradients
        for i, (eg, tg) in enumerate(zip(eager_grads, traced_grads)):
            if eg is not None and isinstance(tg, torch.Tensor):
                # eager grad may be DTensor, get local tensor
                eg_local = eg._local_tensor if hasattr(eg, "_local_tensor") else eg
                # traced grad is flat sharded [numel], eager is shaped — flatten both
                diff = (eg_local.flatten() - tg.flatten()).abs().max().item()
                log.info(
                    f"  grad[{i}]: max_diff={diff:.6e}, "
                    f"eager={eg_local.shape}, traced={tg.shape}"
                )
            else:
                log.info(f"  grad[{i}]: eager={eg}, traced={tg}")


def main():
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        log.info(f"Running with {world_size} GPUs")
        log.info("=" * 60)

    if rank == 0:
        log.info("=" * 60)
        log.info("Test: trace_module with FSDP2")
    test_trace_module()
    dist.barrier()

    if rank == 0:
        log.info("=" * 60)
        log.info("All tests done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
