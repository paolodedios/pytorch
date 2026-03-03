"""CUDA Event abstractions for Inductor stream support.

This module provides event types for synchronizing between CUDA streams when
nodes have user-annotated stream assignments.

Attributes:
    ENTRANCE_EVENT: Name of the first event on the default CUDA Stream that got recorded before all
        kernels.
    EVENT_NAME_TEMPLATE: Python string template to generate event names. Can be used as:

            idx: int = ...
            event = EVENT_NAME_TEMPLATE.format(event_idx=idx)
"""

from __future__ import annotations

import dataclasses
import functools
import itertools

from torch._inductor.codegen.wrapper import IndentedBuffer, WrapperLine


DEFAULT_STREAM: str = "default_stream"
DEFAULT_STREAM_IDX: int = 0
ENTRANCE_EVENT: str = "event0"
EVENT_NAME_TEMPLATE: str = "event{event_idx:d}"
STREAM_NAME_TEMPLATE: str = "stream{stream_idx:d}"


@functools.lru_cache
def get_stream_name(stream_idx: int) -> str:
    """Generate CUDA Stream name from stream index number."""
    if stream_idx == 0:
        return DEFAULT_STREAM
    else:
        return STREAM_NAME_TEMPLATE.format(stream_idx=stream_idx)


@functools.total_ordering
@dataclasses.dataclass
class CudaEventSym:
    """Symbolic representation of CUDA Events in the Inductor scheduling phase.

    Args:
        factory: The CUDAEventFactory that generate this event.
        idx: Indexing number assigned in chronological order during scheduling.
        originate_stream_idx: The index of the CUDA stream that this event originated from.
        materialized_event: The actual CUDA Event name that will be used in the final PyTorch
            program.

    Note:
        In most cases this class should not be used standalone. Use
        `CUDAEventFactory.get_sym_event()` to instantiate one.
    """

    factory: CudaEventFactory
    idx: int
    originate_stream_idx: int
    materialized_event: str | None = None

    def __lt__(self, rhs: CudaEventSym) -> bool:
        """Whether the current event is generated before the rhs event."""
        if self.factory is not rhs.factory:
            return NotImplemented
        return (self.idx, self.originate_stream_idx) < (
            rhs.idx,
            rhs.originate_stream_idx,
        )

    def __eq__(self, rhs: object) -> bool:
        """Whether the current event is identical to the rhs event."""
        if not isinstance(rhs, CudaEventSym):
            return NotImplemented
        return (
            self.idx == rhs.idx
            and self.originate_stream_idx == rhs.originate_stream_idx
            and self.factory is rhs.factory
        )

    def __str__(self) -> str:
        """Represent this symbolic event in string."""
        ret = f"{self.__class__.__name__} (idx={self.idx}"
        ret += f", originate_stream_idx={self.originate_stream_idx}"
        if self.materialized_event:
            ret += f", materialized to `{self.materialized_event}`"
        ret += ")"
        return ret

    def __hash__(self) -> int:
        """Hash this symbolic event."""
        return hash((id(self.factory), self.idx, self.originate_stream_idx))

    def record(self, stream_idx: int) -> _CudaEventRecordLine:
        """Record this event on a given stream."""
        stream = get_stream_name(stream_idx)
        return _CudaEventRecordLine(self, stream)

    def wait(self, stream_idx: int) -> _CudaEventWaitLine:
        """Wait for this event to complete on a given stream."""
        stream = get_stream_name(stream_idx)
        return _CudaEventWaitLine(self, stream)


@dataclasses.dataclass
class _CudaEventRecordLine(WrapperLine):
    event: CudaEventSym
    stream: str

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.event.materialized_event is None
        self.event.materialized_event = self.event.factory.get_materialized_event(code)
        code.writeline(f"{self.event.materialized_event}.record({self.stream})")


@dataclasses.dataclass
class _CudaEventWaitLine(WrapperLine):
    event: CudaEventSym
    stream: str

    def codegen(self, code: IndentedBuffer) -> None:
        assert self.event.materialized_event is not None
        code.writeline(f"{self.event.materialized_event}.wait({self.stream})")


class CudaEventFactory:
    """A factory that manages CUDA event creations and materializations.

    This factory maintains internal states to ensure that created cuda events get monotonically
    increasing indices as compilation goes along.
    """

    def __init__(self) -> None:
        self.symbolic_event_idx: itertools.count = itertools.count(start=1)
        self.materialized_event_idx: itertools.count = itertools.count(start=1)
        self._entrance_event: CudaEventSym | None = None

    def get_entrance_event(self) -> CudaEventSym:
        """Return the cuda event that corresponding to compute graph entering."""
        if self._entrance_event is None:
            self._entrance_event = CudaEventSym(
                factory=self,
                idx=0,
                originate_stream_idx=DEFAULT_STREAM_IDX,
            )
            # Code-gen for entrance event is almost hard-coded in device guard enter so the
            # materialization is slightly different here.
            self._entrance_event.materialized_event = ENTRANCE_EVENT
        return self._entrance_event

    def get_sym_event(self, originate_stream_idx: int) -> CudaEventSym:
        """Allocate a symbolic cuda event."""
        return CudaEventSym(
            factory=self,
            idx=next(self.symbolic_event_idx),
            originate_stream_idx=originate_stream_idx,
        )

    def get_materialized_event(self, code: IndentedBuffer) -> str:
        """Allocate a materialized cuda event."""
        event = EVENT_NAME_TEMPLATE.format(
            event_idx=next(self.materialized_event_idx)
        )
        code.writeline(f"{event} = torch.cuda.Event()")
        return event
