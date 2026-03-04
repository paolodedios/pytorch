"""
Lower functional ops with torch.Tag.out_variant to their .out variant for Inductor memory planning.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch._ops import OpOverload
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .ir import tensor_is_aligned
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)


def try_lower_to_out_variant(
    kernel: OpOverload,
    example_output: Any,
    tensor_args: Sequence[ir.IRNode],
    non_tensor_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Optional[ir.IRNode]:
    """Try lowering a single-output functional op to ExternKernelOut.

    Returns an ExternKernelOut node on success, None on failure.
    Multi-output ops are handled in FallbackKernel.create().
    """
    if not isinstance(kernel, OpOverload):
        return None

    from torch._library._out_variant import (
        _is_functional,
        get_out_arg_names,
        to_out_variant,
    )

    if not _is_functional(kernel._schema):
        return None

    out_op = to_out_variant(kernel)
    if out_op is None:
        return None

    out_arg_names = get_out_arg_names(out_op)

    # Only handle single-output. multi-output is in FallbackKernel.create()
    if not isinstance(example_output, torch.Tensor):
        return None
    if len(out_arg_names) != 1:
        log.debug(
            "Skipping %s: single output but %d out args",
            kernel,
            len(out_arg_names),
        )
        return None
    return _lower_single_output(
        kernel,
        out_op,
        example_output,
        tensor_args,
        non_tensor_args,
        kwargs,
    )


def _lower_single_output(
    kernel: OpOverload,
    out_op: OpOverload,
    example_output: torch.Tensor,
    tensor_args: Sequence[ir.IRNode],
    non_tensor_args: Sequence[Any],
    kwargs: dict[str, Any],
) -> ir.ExternKernelOut:
    """Lower a single-output functional op to ExternKernelOut."""
    layout = ir.FixedLayout(
        device=example_output.device,
        dtype=example_output.dtype,
        size=[*example_output.shape],
        stride=[*example_output.stride()],
    )

    python_kernel_name = _make_python_kernel_name(out_op)

    node = ir.ExternKernelOut(
        layout=layout,
        inputs=list(tensor_args),
        constant_args=list(non_tensor_args),
        kwargs=kwargs,
        python_kernel_name=python_kernel_name,
        op_overload=out_op,
    )

    log.debug("Lowered %s -> %s via ExternKernelOut", kernel, out_op)
    return node


def try_lower_multi_output_to_out_variant(
    kernel: OpOverload,
    example_output: Any,
    packed: ir.FallbackKernel,
    *,
    has_unaligned_input: bool = False,
) -> Optional[Sequence[AllocatingMultiOutput]]:
    """Configure a multi-output FallbackKernel to use .out() codegen.

    Create AllocatingMultiOutput nodes (should_allocate=True) instead of standard
    MultiOutput nodes, enabling Inductor-controlled buffer allocation.
    Returns None if the op doesn't have a matching .out variant.
    """
    if not isinstance(kernel, OpOverload):
        return None

    from torch._library._out_variant import (
        _is_functional,
        get_out_arg_names,
        to_out_variant,
    )

    if not _is_functional(kernel._schema):
        return None

    if not isinstance(example_output, (tuple, list)):
        return None

    out_op = to_out_variant(kernel)
    if out_op is None:
        return None

    out_arg_names = get_out_arg_names(out_op)
    tensors = [t for t in example_output if isinstance(t, torch.Tensor)]
    if len(tensors) != len(example_output) or len(tensors) != len(out_arg_names):
        log.debug(
            "Skipping %s: %d tensor outputs vs %d out args",
            kernel,
            len(tensors),
            len(out_arg_names),
        )
        return None

    # Configure the packed FallbackKernel for .out() codegen
    packed.out_variant_op = out_op
    packed.out_arg_names = out_arg_names
    packed.python_kernel_name = _make_python_kernel_name(out_op)
    packed.op_overload = out_op

    outputs = []
    for i, tensor_out in enumerate(example_output):
        layout = ir.FixedLayout(
            device=tensor_out.device,
            dtype=tensor_out.dtype,
            size=[*tensor_out.shape],
            stride=[*tensor_out.stride()],
        )
        multi_out = AllocatingMultiOutput(
            layout=layout,
            input=packed,
            indices=[(type(example_output), i)],
        )
        if (
            config.assume_unaligned_fallback_output
            or has_unaligned_input
            or not tensor_is_aligned(tensor_out)
        ):
            V.graph.unaligned_buffers.add(multi_out.name)  # type: ignore[arg-type]
        outputs.append(multi_out)

    packed.out_variant_output_nodes = outputs

    log.debug(
        "Lowered %s -> %s (multi-output, %d outputs, out args: %s)",
        kernel,
        out_op,
        len(outputs),
        out_arg_names,
    )
    if isinstance(example_output, tuple):
        return tuple(outputs)  # type: ignore[return-value]
    return list(outputs)


def codegen_multi_output_out_variant(node: ir.FallbackKernel, wrapper: Any) -> None:
    """Emit .out() call with pre-allocated output buffers as kwargs."""
    node.codegen_comment(wrapper)
    kernel_name = node.get_kernel_name()

    for out_node in node.out_variant_output_nodes:
        wrapper.codegen_allocation(out_node)

    args = _codegen_input_args(node)
    kwargs_list = _codegen_kwargs(node, skip_names=OrderedSet(node.out_arg_names))

    all_args = [*args, *kwargs_list]
    for out_name, out_node in zip(node.out_arg_names, node.out_variant_output_nodes):
        all_args.append(f"{out_name}={out_node.get_name()}")

    wrapper.writeline(f"{node.get_name()} = {kernel_name}({', '.join(all_args)})")

    for out_node in node.out_variant_output_nodes:
        if isinstance(out_node.layout, ir.Layout):
            out_node.codegen_size_asserts(wrapper)


def _make_python_kernel_name(out_op: OpOverload) -> str:
    """Build fully-qualified kernel name, e.g. 'torch.ops.mylib.add_one.out'."""
    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"torch.ops.{ns}.{op_name}.{overload}"


def _codegen_input_args(node: ir.ExternKernel) -> list[str]:
    """Codegen positional + constant args, excluding out args."""
    assert ir.is_node_sequence(node.inputs)
    args = [x.codegen_reference() for x in node.inputs]  # type: ignore[union-attr]
    for const in node.constant_args:
        args.append(V.graph.wrapper_code.val_to_arg_str(const))
    return args


def _codegen_kwargs(node: ir.ExternKernel, skip_names: OrderedSet[str]) -> list[str]:
    """Codegen keyword args, skipping out arg names (appended separately)."""
    result = []
    for k, v in node.kwargs.items():
        if k not in skip_names:
            result.append(f"{k}={V.graph.wrapper_code.val_to_arg_str(v)}")
    return result


class AllocatingMultiOutput(ir.MultiOutput):
    """MultiOutput with Inductor-controlled allocation for buffer reuse.

    Overrides should_allocate()=True for buffer planning, and skips
    tuple-indexing codegen since .out() writes directly into these buffers.
    """

    def should_allocate(self) -> bool:
        return True

    def codegen(self, wrapper: Any) -> None:
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)
