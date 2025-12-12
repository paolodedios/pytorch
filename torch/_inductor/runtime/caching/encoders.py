# pyre-strict

"""
Custom encoder functions for use with PersistentMemoizer.

This module provides reusable encoder functions that convert function parameters
into JSON-serializable dictionaries for caching purposes.
"""

from typing import Any

import torch
from torch import Tensor


def encode_tensor(t: Tensor) -> dict[str, Any]:
    """Encode a tensor's metadata into a JSON-serializable dict.
    
    Args:
        t: PyTorch tensor to encode
        
    Returns:
        Dict containing shape, stride, and dtype information
    """
    return {
        "shape": tuple(t.shape),
        "stride": tuple(t.stride()),
        "dtype": str(t.dtype),
    }


def should_pad_params_encoder(
    match: Any,  # Match type from pattern_matcher
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> dict[str, Any]:
    """Encode parameters for _should_pad into a human-readable dict.

    This encoder extracts only the information needed for caching:
    - Tensor shape, stride, and dtype (not the actual data)
    - Whether padding time should be excluded for mat1 and mat2
    - The operation as a string

    Args:
        match: The pattern match object
        mat1: First matrix tensor
        mat2: Second matrix tensor
        op: The operation being performed
        input: Optional input tensor for addmm

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    # Import here to avoid circular dependency
    from torch._inductor.fx_passes.pad_mm import should_exclude_padding_time

    return {
        "mat1": encode_tensor(mat1),
        "mat2": encode_tensor(mat2),
        "op": str(op),
        "input": encode_tensor(input) if input is not None else None,
        "mat1_exclude_padding_time": should_exclude_padding_time(match, "mat1"),
        "mat2_exclude_padding_time": should_exclude_padding_time(match, "mat2"),
    }
