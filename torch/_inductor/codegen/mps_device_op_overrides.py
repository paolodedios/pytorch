from __future__ import annotations

from .common import DeviceIndexExpr, DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    # MPS is single-device; the device index is rendered but not inspected (that
    # invariant belongs at device selection, not in this code-formatting method).
    def device_guard(self, device_idx: DeviceIndexExpr) -> str:
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: DeviceIndexExpr) -> str:
        return "pass  # MPS set device"

    def kernel_driver(self) -> str:
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        return "MTLFunction_t"


register_device_op_overrides("mps", MPSDeviceOpOverrides())
