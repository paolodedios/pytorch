"""Custom ops for async activation offloading between GPU and CPU.

These ops encapsulate stream management internally, producing a clean 2-node
IR pattern (offload/reload + wait_tensor) similar to c10d functional collectives.

Streams and events are passed as integer indices into the graph's external
object table, ensuring they are created once and reused across iterations.
Synchronization details are hidden inside a module-level registry keyed on
``data_ptr()``, so ``ao.wait_tensor`` takes only the tensor itself (plus an
optional keepalive).

Offload pattern:
    async_cpu = ao.offload(transfer_stream_idx, completion_event_idx, gpu_tensor)
    cpu_tensor = ao.wait_tensor(async_cpu, gpu_tensor)
                                (keepalive arg extends gpu_tensor lifetime past the async D2H copy)

Reload pattern:
    async_gpu = ao.reload(transfer_stream_idx, completion_event_idx, start_event_idx, cpu_tensor, device)
    gpu_tensor = ao.wait_tensor(async_gpu)
"""

import torch
from torch._dynamo.variables.streams import _get_event_by_index, _get_stream_by_index
from torch._library.custom_ops import custom_op
from torch.fx import has_side_effect


# Maps data_ptr() -> (completion_event, device) for tensors produced by
# ao.offload / ao.reload.  Consumed (popped) by ao.wait_tensor.
# Not thread-safe — graph execution is single-threaded Python.
_wait_registry: dict[int, tuple[torch.Event, torch.device]] = {}


def _register_wait(
    tensor: torch.Tensor, completion_event: torch.Event, device: torch.device
) -> None:
    _wait_registry[tensor.data_ptr()] = (completion_event, device)


def _pop_wait(tensor: torch.Tensor) -> tuple[torch.Event, torch.device]:
    key = tensor.data_ptr()
    try:
        return _wait_registry.pop(key)
    except KeyError:
        raise RuntimeError(
            f"ao.wait_tensor: no pending transfer for tensor with data_ptr={key}. "
            "Every ao.wait_tensor must be paired with a preceding ao.offload or ao.reload."
        ) from None


def _clear_wait_registry() -> None:
    _wait_registry.clear()


@custom_op("ao::offload", mutates_args=())
def offload(
    transfer_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    """Async offload a GPU tensor to CPU on a dedicated transfer stream.

    Callers MUST pair this with an ``ao.wait_tensor`` that passes the source GPU
    tensor as ``keepalive`` to extend its lifetime past the async D2H copy.
    Do NOT use ``record_stream`` — it causes memory fragmentation and
    unbounded memory growth.

    Uses pinned-memory allocation + copy_ so the transfer is compatible
    with CUDA graph capture.
    """
    transfer_stream = _get_stream_by_index(transfer_stream_idx)
    completion_event = _get_event_by_index(completion_event_idx)
    current_stream = torch.accelerator.current_stream(tensor.device)

    transfer_stream.wait_stream(current_stream)

    torch.accelerator.set_stream(transfer_stream)
    result = torch.empty_like(tensor, device="cpu", pin_memory=True)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    _register_wait(result, completion_event, tensor.device)

    return result


@offload.register_fake
def _(
    transfer_stream_idx: int, completion_event_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(tensor, device="cpu")


@custom_op("ao::reload", mutates_args=())
def reload(
    transfer_stream_idx: int,
    completion_event_idx: int,
    start_event_idx: int | None,
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Async reload a CPU tensor to GPU on a dedicated transfer stream.

    The GPU tensor is allocated on the compute stream to avoid cross-stream
    allocator ownership issues. The H2D copy runs on the transfer stream.

    When ``start_event_idx`` is not None, the transfer stream waits on
    that event before beginning the H2D copy.  A ``streams::record_event``
    node placed at the desired point in the backward graph records this
    event on the compute stream, gating when the transfer actually starts.
    Pass None to skip the wait (transfer begins immediately).
    """
    transfer_stream = _get_stream_by_index(transfer_stream_idx)
    completion_event = _get_event_by_index(completion_event_idx)
    current_stream = torch.accelerator.current_stream(device)

    if start_event_idx is not None:
        start_event = _get_event_by_index(start_event_idx)
        transfer_stream.wait_event(start_event)

    # Allocate on compute stream so the allocator tracks ownership correctly
    result = torch.empty_like(tensor, device=device)

    torch.accelerator.set_stream(transfer_stream)
    result.copy_(tensor, non_blocking=True)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    _register_wait(result, completion_event, device)

    return result


@reload.register_fake
def _(
    transfer_stream_idx: int,
    completion_event_idx: int,
    start_event_idx: int | None,
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty_like(tensor, device=device)


# ao::wait_tensor is defined via torch.library with an aliasing schema so the
# output can alias the input (custom_op forbids this).
#
# Uses CompositeExplicitAutograd (single impl for all devices) because the
# offload case has mixed-device args: ``tensor`` is CPU (the offload result)
# while ``keepalive`` is CUDA (the source GPU tensor). A single impl avoids
# relying on device-priority dispatch ordering.
#
# Synchronization details (completion event, device) are looked up from
# ``_wait_registry`` keyed on ``tensor.data_ptr()``, so callers don't need
# to pass stream/event indices — matching the c10d::wait_tensor pattern.
#
# ``keepalive`` is not read by the op — its sole purpose is to create a graph
# dependency that extends the tensor's lifetime in the FX graph. For offload,
# this keeps the source GPU tensor alive until the compute stream has waited
# on the D2H completion event, preventing the allocator from reclaiming it
# while the async copy is still in flight.
_lib = torch.library.Library("ao", "DEF")
_lib.define("wait_tensor(Tensor(a) tensor, Tensor? keepalive=None) -> Tensor(a)")


@torch.library.impl("ao::wait_tensor", "CompositeExplicitAutograd")
def _ao_wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
) -> torch.Tensor:
    completion_event, device = _pop_wait(tensor)
    current_stream = torch.accelerator.current_stream(device)
    current_stream.wait_event(completion_event)
    return tensor


@torch.library.register_fake("ao::wait_tensor")
def _ao_wait_tensor_fake(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.ao.wait_tensor.default)


def wait_tensor(
    tensor: torch.Tensor,
    keepalive: torch.Tensor | None = None,
) -> torch.Tensor:
    """Callable wrapper so ``wait_tensor`` can be imported by name for op registration."""
    return torch.ops.ao.wait_tensor.default(tensor, keepalive)
