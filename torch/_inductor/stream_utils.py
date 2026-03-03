"""Stream utilities for Inductor codegen.

This module provides utilities for managing CUDA streams in Inductor when
nodes have user-annotated stream assignments. It includes constants for
stream/event naming.
"""

from __future__ import annotations

import functools


__all__ = [
    "DEFAULT_STREAM",
    "DEFAULT_STREAM_IDX",
    "ENTRANCE_EVENT",
    "EVENT_NAME_TEMPLATE",
    "STREAM_NAME_TEMPLATE",
    "get_stream_name",
]

DEFAULT_STREAM: str = "default_stream"
DEFAULT_STREAM_IDX: int = 0
ENTRANCE_EVENT: str = "event0"
EVENT_NAME_TEMPLATE: str = "event{event_idx:d}"
STREAM_NAME_TEMPLATE: str = "stream{stream_idx:d}"


@functools.lru_cache
def get_stream_name(stream_idx: int) -> str:
    """Generate CUDA Stream name from stream index number.

    Args:
        stream_idx: Non-negative index number. 0 refers to the default stream, others refer to side
            streams.
    """
    if stream_idx == 0:
        return DEFAULT_STREAM
    else:
        return STREAM_NAME_TEMPLATE.format(stream_idx=stream_idx)
