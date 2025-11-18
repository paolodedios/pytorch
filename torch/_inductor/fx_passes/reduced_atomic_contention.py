"""
Partitioned scatter optimization for index_add/index_put operations.

This optimization reduces atomic contention by distributing scatter operations
across partitions of an expanded buffers, then reducing the results.

Algorithm:
1. Compute partition assignments: partition_id = thread_id % num_partitions
2. Create expanded buffers: size = num_partitions × original_size
3. Adjust indices: adjusted_idx = original_idx + (partition_id × partition_size)
4. Perform partitioned scatter with reduced contention
5. Reduce across partitions: sum(partitions, dim=0)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.fx as fx
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

log = logging.getLogger(__name__)

_index_scatter_patterns = PatternMatcherPass()


class PartitionConfig:
    """Configuration and heuristics for partition-based scatter optimization."""

    MIN_PARTITIONS = 2
    MAX_PARTITIONS = 128

    @staticmethod
    def estimate_optimal_partitions(output_size: int, index_size: int) -> int:
        """Estimate optimal partitions: 
        
        Note: This heuristic assumes the index is uniformly distributed across the output.
        The contention ratio will be higher if the index is not uniformly distributed.
        """
        contention_ratio = index_size / output_size
        base = 2 if contention_ratio < 0.5 else int(math.sqrt(contention_ratio) * 8)
        partitions = 2 ** math.ceil(math.log2(max(2, base)))
        return min(PartitionConfig.MAX_PARTITIONS, partitions)

    @staticmethod
    def should_optimize(
        output_size: int,
        index_size: int,
        num_partitions: int,
        element_bytes: int = 4,
        available_memory: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Determine if optimization should be applied."""
        if output_size == 0 or index_size == 0:
            return False, "Empty tensor"

        contention_ratio = index_size / output_size
        if contention_ratio < 0.05:
            return False, f"Low contention (ratio={contention_ratio:.3f})"

        if available_memory is not None:
            required = output_size * element_bytes * num_partitions
            if required > available_memory * 0.1:
                return False, f"Insufficient memory: {required/1e9:.2f}GB > {available_memory*0.1/1e9:.2f}GB"

        return True, "Optimization applicable"


def extract_tensor_metadata(node: fx.Node) -> Optional[dict[str, Any]]:
    """Extract shape, dtype, and device from FX node."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None
    
    val = node.meta["val"]
    if not (isinstance(val, torch.Tensor) or (hasattr(val, "__class__") and "Tensor" in val.__class__.__name__)):
        return None
    
    return {
        "shape": tuple(val.shape) if hasattr(val, "shape") else None,
        "dtype": val.dtype if hasattr(val, "dtype") else None,
        "device": val.device if hasattr(val, "device") else None,
        "numel": val.numel() if hasattr(val, "numel") else None,
    }


def get_index_scatter_metadata(match: Match) -> Optional[dict[str, Any]]:
    """Extract metadata from matched index_put pattern."""
    try:
        output_node = match.output_node()
        if not output_node or not hasattr(output_node, "args") or len(output_node.args) < 3:
            return None

        args = output_node.args
        input_node = args[0]
        indices_arg = args[1]
        index_node = indices_arg[0] if isinstance(indices_arg, (list, tuple)) and indices_arg else indices_arg

        output_meta = extract_tensor_metadata(input_node)
        index_meta = extract_tensor_metadata(index_node)

        if not output_meta or not index_meta:
            return None

        return {
            "output_size": output_meta["numel"],
            "index_size": index_meta["numel"],
            "output_shape": output_meta["shape"],
            "index_shape": index_meta["shape"],
            "dim": 0,
            "dtype": output_meta["dtype"],
            "device": output_meta["device"],
        }
    except Exception as e:
        log.debug(f"Error extracting metadata: {e}")
        return None


def validate_match(match: Match) -> bool:
    """Validate if pattern match should be optimized."""
    output_node = match.output_node()
    if not output_node or not hasattr(output_node, "args") or len(output_node.args) < 4:
        return False

    # Only apply pass if we are accumulating (accumulate=True)
    if output_node.args[3] is not True:
        return False

    metadata = get_index_scatter_metadata(match)
    if not metadata:
        return False

    element_bytes = metadata["dtype"].itemsize if metadata["dtype"] else 4

    # Estimate optimal partition count
    estimated_partitions = PartitionConfig.estimate_optimal_partitions(
        metadata["output_size"], metadata["index_size"]
    )

    # Check memory constraints and scale down if needed
    if torch.cuda.is_available():
        try:
            _, total_memory = torch.cuda.mem_get_info()
            memory_budget = total_memory * 0.10
            
            while estimated_partitions >= PartitionConfig.MIN_PARTITIONS:
                required = metadata["output_size"] * element_bytes * estimated_partitions
                if required <= memory_budget:
                    break
                estimated_partitions //= 2
            else:
                return False
        except Exception:
            pass

    # Final validation check
    should_apply, reason = PartitionConfig.should_optimize(
        metadata["output_size"],
        metadata["index_size"],
        estimated_partitions,
        element_bytes,
    )
    
    if not should_apply:
        log.debug(f"Skipping partitioned scatter optimization: {reason}")
        return False

    # Store the adjusted partition count on the match for use in replacement
    match._adjusted_num_partitions = estimated_partitions
    return True


def create_replacement(match: Match, input_tensor: fx.Node, indices: Any, values: fx.Node):
    """Replace high-contention index_put with partitioned scatter."""
    graph = match.graph
    matched_node = match.output_node()
    
    # Retrieve the partition count we computed during validation
    actual_num_partitions = getattr(match, "_adjusted_num_partitions", PartitionConfig.MAX_PARTITIONS // 2)

    # Extract index from list
    index = indices[0] if isinstance(indices, (list, tuple)) and indices else indices

    input_meta = input_tensor.meta.get("val")
    index_meta = index.meta.get("val")
    values_meta = values.meta.get("val")

    dim = 0
    dim_size = input_meta.shape[dim]
    N = index_meta.shape[0]
    device = index_meta.device

    with graph.inserting_before(matched_node):
        # Generate sequence of ints with iota, and then compute partition IDs (thread_id % num_partitions)
        thread_ids = graph.call_function(
            torch.ops.prims.iota.default,
            args=(N,),
            kwargs={
                "start": 0,
                "step": 1,
                "dtype": torch.int64,
                "device": device,
                "requires_grad": False,
            },
        )
        partition_ids = graph.call_function(
            torch.ops.aten.bitwise_and.Scalar,
            args=(thread_ids, actual_num_partitions - 1),
        )

        # Create expanded buffer - original buffer replicated num_partitions times
        expanded_shape = list(input_meta.shape)
        expanded_shape[dim] *= actual_num_partitions
        expanded_buffer = graph.call_function(
            torch.ops.aten.full.default,
            args=(expanded_shape, 0),
            kwargs={
                "dtype": values_meta.dtype,
                "layout": torch.strided,
                "device": device,
                "pin_memory": False,
            },
        )

        # Adjust indices: adjusted_idx = original_idx + (partition_id × dim_size)
        partition_offsets = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(partition_ids, dim_size)
        )
        adjusted_indices = graph.call_function(
            torch.ops.aten.add.Tensor, args=(index, partition_offsets)
        )

        # Perform scatter operation on expanded buffer
        scattered_buffer = graph.call_function(
            torch.ops.aten.index_put.default,
            args=(expanded_buffer, [adjusted_indices], values, True),
        )

        # Reduce across partitions
        reduce_shape = [actual_num_partitions, dim_size] + list(expanded_shape[1:])
        reshaped = graph.call_function(
            torch.ops.aten.view.default, args=(scattered_buffer, reduce_shape)
        )
        result = graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(reshaped, [0])
        )

        # Add to original input
        output = graph.call_function(
            torch.ops.aten.add.Tensor, args=(input_tensor, result)
        )

    matched_node.replace_all_uses_with(output)
    graph.erase_node(matched_node)
    return output


def register_partitioned_scatter_patterns() -> None:
    """Register patterns for both index_put and index_put_ (in-place)."""
    
    # Register pattern for out-of-place index_put
    register_graph_pattern(
        CallFunction(torch.ops.aten.index_put.default, Arg(), Arg(), Arg(), True),
        extra_check=validate_match,
        pass_dict=_index_scatter_patterns,
    )(create_replacement)

    # Register pattern for in-place index_put_
    register_graph_pattern(
        CallFunction(torch.ops.aten.index_put_.default, Arg(), Arg(), Arg(), True),
        extra_check=validate_match,
        pass_dict=_index_scatter_patterns,
    )(create_replacement)


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    """Apply partitioned scatter optimization to high-contention index_put operations."""
    if not getattr(config, "partitioned_scatter_enabled", False):
        return graph

    if not _index_scatter_patterns.patterns:
        register_partitioned_scatter_patterns()

    num_matches = _index_scatter_patterns.apply(graph)  # Changed from gm.graph to graph
    if num_matches > 0:
        log.info(f"Applied partitioned scatter optimization to {num_matches} operations")
        graph.lint()  # Changed from gm.graph.lint() to graph.lint()

    return graph


__all__ = [
    "partitioned_scatter_optimization_pass",
    "PartitionConfig",
]
