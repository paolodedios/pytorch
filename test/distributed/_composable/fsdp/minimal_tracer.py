"""
Minimal make_fx based tracer with:
- FSDP internal parameter lifting (_sharded_param_data as graph inputs)
- FSDP hooks use fsdp::unshard_ custom op (set via _make_fx_tracing flag)
- Tensor subclass input/output support (e.g., DTensor)
- Fake mode tracing

Cloned from sixlib/sixlib/minimal_tracer.py for FSDP2 tracing experiments.
"""

import contextlib
import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp._fully_shard._fsdp_common import _make_fx_tracing
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


# =============================================
# FSDP utilities
# =============================================


@dataclass
class FSDPParamMeta:
    """Metadata for one FSDP-managed parameter."""

    fqn: str
    orig_size: list[int]
    orig_stride: list[int]
    group_size: int


def _collect_fsdp_params_and_metas(
    mod: nn.Module,
) -> tuple[list[FSDPParam], list[FSDPParamMeta]]:
    """Collect FSDPParam objects and their metadata from FSDP-wrapped modules."""
    fsdp_params = []
    param_metas = []
    for module in mod.modules():
        state = _get_module_state(module)
        if isinstance(state, FSDPState) and state._fsdp_param_group is not None:
            pg = state._fsdp_param_group
            group_size = pg._all_gather_process_group.size()
            for fp in pg.fsdp_params:
                fsdp_params.append(fp)
                param_metas.append(
                    FSDPParamMeta(
                        fqn=fp._param_fqn,
                        orig_size=list(fp._orig_size),
                        orig_stride=list(fp._contiguous_orig_stride),
                        group_size=group_size,
                    )
                )
    return fsdp_params, param_metas


@contextmanager
def _enable_make_fx_tracing():
    """Enable make_fx tracing mode for FSDP hooks."""
    import torch.distributed.fsdp._fully_shard._fsdp_common as fsdp_common

    original = fsdp_common._make_fx_tracing
    fsdp_common._make_fx_tracing = True
    try:
        yield
    finally:
        fsdp_common._make_fx_tracing = original


@contextmanager
def _swap_sharded_param_data(
    fsdp_params: list[FSDPParam],
    new_sharded_datas: list[torch.Tensor],
):
    """Temporarily swap _sharded_param_data on FSDPParam objects."""
    originals = []
    for fsdp_param, new_data in zip(fsdp_params, new_sharded_datas):
        originals.append(fsdp_param._sharded_param_data)
        fsdp_param._sharded_param_data = new_data
    try:
        yield
    finally:
        for fsdp_param, orig in zip(fsdp_params, originals):
            fsdp_param._sharded_param_data = orig


# =============================================
# Subclass handling
# =============================================


@dataclass
class SubclassMeta:
    cls: type
    attrs: list[str]
    ctx: Any
    inner_metas: dict[str, tuple[int, Any]]
    outer_size: torch.Size
    outer_stride: tuple[int, ...]


def unwrap_subclass(
    t: torch.Tensor,
) -> tuple[list[torch.Tensor], SubclassMeta | None]:
    if not is_traceable_wrapper_subclass(t):
        return [t], None
    attrs, ctx = t.__tensor_flatten__()
    all_inner = []
    inner_metas = {}
    for attr in attrs:
        inner_t = getattr(t, attr)
        tensors, meta = unwrap_subclass(inner_t)
        all_inner.extend(tensors)
        inner_metas[attr] = (len(tensors), meta)
    meta = SubclassMeta(
        cls=type(t), attrs=attrs, ctx=ctx, inner_metas=inner_metas,
        outer_size=t.size(), outer_stride=t.stride(),
    )
    return all_inner, meta


def wrap_to_subclass(
    plain_tensors: list[torch.Tensor], meta: SubclassMeta
) -> torch.Tensor:
    inner_dict = {}
    idx = 0
    for attr in meta.attrs:
        num_inner, inner_meta = meta.inner_metas[attr]
        inner_tensors = plain_tensors[idx : idx + num_inner]
        idx += num_inner
        if inner_meta is None:
            inner_dict[attr] = inner_tensors[0]
        else:
            inner_dict[attr] = wrap_to_subclass(list(inner_tensors), inner_meta)
    return meta.cls.__tensor_unflatten__(
        inner_dict, meta.ctx, meta.outer_size, meta.outer_stride
    )


def wrap_inputs_to_subclasses(
    plain_args: tuple[torch.Tensor, ...],
    subclass_metas: list[tuple[int, SubclassMeta | None]],
) -> list[torch.Tensor]:
    wrapped = []
    idx = 0
    for num_tensors, meta in subclass_metas:
        tensors = plain_args[idx : idx + num_tensors]
        idx += num_tensors
        if meta is None:
            wrapped.append(tensors[0])
        else:
            wrapped.append(wrap_to_subclass(list(tensors), meta))
    return wrapped


# =============================================
# Graph cleanup
# =============================================


def _remove_cpu_shadow_chains(gm: torch.fx.GraphModule) -> None:
    to_remove: set[torch.fx.Node] = set()
    for node in gm.graph.nodes:
        if node in to_remove:
            continue
        if not (
            node.op == "call_function"
            and node.target == torch.ops.aten.empty_strided.default
        ):
            continue
        device = node.kwargs.get("device")
        if device is None or device.type != "cpu":
            continue
        chain: set[torch.fx.Node] = set()
        queue = [node]
        feeds_gpu = False
        while queue and not feeds_gpu:
            current = queue.pop()
            if current in chain:
                continue
            chain.add(current)
            for user in current.users:
                val = user.meta.get("val")
                if isinstance(val, torch.Tensor) and val.device.type != "cpu":
                    if user.users:
                        feeds_gpu = True
                        break
                    chain.add(user)
                    continue
                queue.append(user)
        if not feeds_gpu:
            to_remove |= chain
    for node in reversed(list(gm.graph.nodes)):
        if node in to_remove:
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()


# =============================================
# Main tracer
# =============================================


def trace_module(
    mod: nn.Module,
    args: tuple,
) -> torch.fx.GraphModule:
    """
    Trace an FSDP2-wrapped module with make_fx.

    Strategy:
    1. Lift FSDP's _sharded_param_data as graph inputs
    2. Enable _make_fx_tracing flag so FSDP hooks use fsdp::unshard_ custom op
    3. Swap lifted sharded data into FSDPParam objects so hooks use graph inputs
    4. FSDP hooks fire naturally (model(x) pattern), but use the custom op path
    5. Sharded data is passed as extra args to mod.forward so user code can
       call autograd.grad w.r.t. them (gradient flows through fsdp::unshard
       backward → fsdp::reduce_scatter_grad)

    Args:
        mod: The nn.Module to trace (must be FSDP-wrapped with lazy init done)
        args: Tuple of input arguments to the module's forward method

    Returns:
        traced: A traced fx.GraphModule
    """

    # ========================================
    # Step 1: Collect FSDP internal sharded param data
    # ========================================
    fsdp_params, param_metas = _collect_fsdp_params_and_metas(mod)
    # Clone sharded data with requires_grad so fsdp::unshard output has grad_fn
    # and autograd.grad can backprop through it
    sharded_datas = []
    for fp in fsdp_params:
        data = fp._sharded_param_data
        if fp.sharded_param.requires_grad:
            data = data.detach().requires_grad_(True)
        sharded_datas.append(data)
    n_fsdp_params = len(sharded_datas)

    # ========================================
    # Step 2: Create functional call wrapper
    # ========================================
    def functional_call(*all_args):
        fsdp_sharded = list(all_args[:n_fsdp_params])
        user_args = all_args[n_fsdp_params:]

        # Swap sharded data into FSDP params so hooks use graph inputs,
        # and enable tracing flag so hooks use the custom op path.
        # Pass fsdp_sharded as extra kwarg so the module can use them
        # as targets for autograd.grad.
        with (
            _swap_sharded_param_data(fsdp_params, fsdp_sharded),
            _enable_make_fx_tracing(),
        ):
            return mod.forward(*user_args, fsdp_sharded_params=fsdp_sharded)

    # Pytree-flatten user args
    user_args_flat, user_args_spec = pytree.tree_flatten(args)
    full_args = tuple(sharded_datas) + tuple(user_args_flat)

    # ========================================
    # Step 3: Detect & unwrap subclass inputs
    # ========================================
    unwrapped_args = []
    subclass_metas = []

    for arg in full_args:
        if isinstance(arg, torch.Tensor) and is_traceable_wrapper_subclass(arg):
            inner_tensors, meta = unwrap_subclass(arg)
            unwrapped_args.extend(inner_tensors)
            subclass_metas.append((len(inner_tensors), meta))
        else:
            unwrapped_args.append(arg)
            subclass_metas.append((1, None))

    # ========================================
    # Step 4: Convert to fake tensors
    # ========================================
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def to_fake(t):
        if isinstance(t, torch.Tensor):
            return fake_mode.from_tensor(t)
        return t

    fake_args = tuple(to_fake(a) for a in unwrapped_args)

    # ========================================
    # Step 5: Create fn that handles subclass wrap/unwrap
    # ========================================
    output_subclass_metas = []

    def fn_with_subclass_handling(*plain_args):
        nonlocal output_subclass_metas
        output_subclass_metas = []

        wrapped_args = wrap_inputs_to_subclasses(plain_args, subclass_metas)
        fsdp_args = wrapped_args[:n_fsdp_params]
        user_args_wrapped = wrapped_args[n_fsdp_params:]
        user_args_restored = pytree.tree_unflatten(
            list(user_args_wrapped), user_args_spec
        )

        outputs = functional_call(*fsdp_args, *user_args_restored)

        flat_outputs, _ = pytree.tree_flatten(outputs)
        unwrapped_outputs = []
        for out in flat_outputs:
            if isinstance(out, torch.Tensor) and is_traceable_wrapper_subclass(out):
                inner, meta = unwrap_subclass(out)
                unwrapped_outputs.extend(inner)
                output_subclass_metas.append((len(inner), meta))
            else:
                unwrapped_outputs.append(out)
                output_subclass_metas.append((1, None))

        return unwrapped_outputs

    # ========================================
    # Step 6: Trace with make_fx under fake mode
    # ========================================
    with fake_mode, preserve_node_meta():
        traced = make_fx(fn_with_subclass_handling, record_stack_traces=True)(
            *fake_args
        )

    # ========================================
    # Step 6.5: Remove CPU shadow nodes
    # ========================================
    _remove_cpu_shadow_chains(traced)

    # ========================================
    # Step 7: Attach metadata for runtime
    # ========================================
    traced._n_fsdp_params = n_fsdp_params
    traced._param_metas = param_metas
    traced._input_subclass_metas = subclass_metas
    traced._output_subclass_metas = output_subclass_metas

    return traced


def run_traced_module(
    traced: torch.fx.GraphModule,
    mod: nn.Module,
    args: tuple,
) -> list[torch.Tensor]:
    """Run traced graph with FSDP sharded params."""
    fsdp_params, _ = _collect_fsdp_params_and_metas(mod)
    sharded_datas = [fp._sharded_param_data for fp in fsdp_params]
    user_args_flat, _ = pytree.tree_flatten(args)

    all_args = []
    for a in itertools.chain(sharded_datas, user_args_flat):
        if isinstance(a, torch.Tensor) and is_traceable_wrapper_subclass(a):
            inner, _ = unwrap_subclass(a)
            all_args.extend(inner)
        else:
            all_args.append(a)

    return traced(*all_args)
