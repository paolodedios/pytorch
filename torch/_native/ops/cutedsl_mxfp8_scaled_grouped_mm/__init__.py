from .. import cu
from .scaled_grouped_mm_mxfp8 import scaled_grouped_mm_mxfp8_register_kernels

# No runtime_available() guard needed here: register_op() already checks
# runtime availability and TORCH_DISABLE_NATIVE_JIT internally.
# The unconditional import of scaled_grouped_mm_mxfp8 is safe because it
# lazily imports DSL runtimes (cutlass, tvm_ffi, cuda.bindings) inside
# functions rather than at module level.
cu.register_op(scaled_grouped_mm_mxfp8_register_kernels)
