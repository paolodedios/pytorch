"""
Codegen for AOTDispatchSubclassWrapper.

Generates a Python function that replaces the data-driven
runtime_unwrap_tensor_subclasses / wrap_tensor_subclasses loop with
a straight-line function where all metadata (indices, attr names,
subclass types, symint positions) is baked in at compile time.
"""

import contextlib
import functools
import keyword
import logging
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import SymInt

from .schemas import OpaqueMeta, PlainTensorMeta, SubclassCreationMeta


log = logging.getLogger(__name__)


def _is_symint_placeholder(x: None | int | SymInt) -> bool:
    """Check whether a size/stride entry is symbolic and needs a runtime value.

    Works both before make_runtime_safe() (entries are SymInt) and after
    (symbolic entries replaced with None, nested ints with -1).
    """
    if x is None:
        return True
    if isinstance(x, SymInt) and not x.node.is_nested_int():
        return True
    return False


def _compute_placeholders(outer: Iterable[None | int | SymInt]) -> list[bool]:
    return [_is_symint_placeholder(s) for s in outer]


def _safe_attr_access(var: str, attr: str) -> str:
    if attr.isidentifier() and not keyword.iskeyword(attr):
        return f"{var}.{attr}"
    return f"getattr({var}, {attr!r})"


class PySourceBuilder:
    """Builds indented Python source for compile/exec, along with the globals
    the generated code closes over and a monotonic fresh-name counter.

    Body lines are written WITHOUT leading whitespace; indentation is managed
    by the ``indent()`` context manager so call sites read as plain code. Pass
    ``fn_name``/``artifact_name`` to emit a ``def`` header and enable
    ``build()``, which routes through _compile_and_exec_source.
    """

    def __init__(
        self,
        fn_name: str | None = None,
        *,
        args: str = "args",
        artifact_name: str | None = None,
    ) -> None:
        self.lines: list[str] = []
        self.globals: dict[str, object] = {}
        self._name_counter: int = 0
        self._indent: int = 0
        self._fn_name = fn_name
        self._artifact_name = artifact_name
        if fn_name is not None:
            self.writeline(f"def {fn_name}({args}):")

    @contextlib.contextmanager
    def indent(self, offset: int = 1) -> Iterator[None]:
        self._indent += offset
        try:
            yield
        finally:
            self._indent -= offset

    def writeline(self, line: str) -> None:
        self.lines.append("    " * self._indent + line)

    def fresh_name(self, prefix: str) -> str:
        name = f"{prefix}_{self._name_counter}"
        self._name_counter += 1
        return name

    def bind(self, **values: object) -> None:
        """Bind live objects into the exec globals by keyword, by reference."""
        self.globals.update(values)

    def bind_value(self, prefix: str, value: object) -> str:
        """Bind a value under a fresh unique name and return that name."""
        name = self.fresh_name(prefix)
        self.globals[name] = value
        return name

    def getvalue(self) -> str:
        return "\n".join(self.lines)

    def build(
        self, *, wrapped_fn: Callable[..., object] | None = None
    ) -> Callable[..., object]:
        assert self._fn_name is not None and self._artifact_name is not None, (  # noqa: S101
            "build() requires fn_name and artifact_name"
        )
        return _compile_and_exec_source(
            self.getvalue(),
            self.globals,
            self._fn_name,
            self._artifact_name,
            wrapped_fn=wrapped_fn,
        )


def _codegen_unwrap_subclass(
    buf: PySourceBuilder,
    meta: SubclassCreationMeta,
    var: str,
    include_symints: bool = True,
) -> None:
    """Emit code to recursively unwrap a single subclass input."""
    for attr, attr_meta in meta.attrs.items():
        match attr_meta:
            case PlainTensorMeta() | OpaqueMeta():
                buf.writeline(f"unwrapped_args.append({_safe_attr_access(var, attr)})")
            case SubclassCreationMeta():
                inner_var = buf.fresh_name("_inner")
                buf.writeline(f"{inner_var} = {_safe_attr_access(var, attr)}")
                _codegen_unwrap_subclass(
                    buf, attr_meta, inner_var, include_symints=include_symints
                )

    # Emit symint extraction
    if include_symints:
        size_placeholders = _compute_placeholders(meta.outer_size)
        stride_placeholders = _compute_placeholders(meta.outer_stride)
        has_size_symints = any(size_placeholders)
        has_stride_symints = any(stride_placeholders)

        if has_size_symints or has_stride_symints:
            size_var = buf.fresh_name("_size")
            buf.writeline(f"{size_var} = {var}.size()")
            for i, is_sym in enumerate(size_placeholders):
                if is_sym:
                    buf.writeline(f"unwrapped_args.append({size_var}[{i}])")

            stride_var = buf.fresh_name("_stride")
            buf.writeline(f"{stride_var} = {var}.stride()")
            for i, is_sym in enumerate(stride_placeholders):
                if is_sym:
                    buf.writeline(f"unwrapped_args.append({stride_var}[{i}])")


def _concrete_value(val: None | int | SymInt) -> int:
    """Get the concrete int value for a non-symbolic size/stride entry.

    Used for entries that are NOT symbolic placeholders, meaning they are
    concrete ints or nested ints (represented as -1 after make_runtime_safe).
    """
    if isinstance(val, int):
        return val
    # Before make_runtime_safe: nested ints are SymInts; use -1 as dummy.
    # After make_runtime_safe: they're already -1.
    if isinstance(val, SymInt) and val.node.is_nested_int():
        return -1
    raise AssertionError(f"Expected concrete int, got {type(val)}: {val}")


def _codegen_wrap_subclass(
    buf: PySourceBuilder,
    meta: SubclassCreationMeta,
    out_idx_ref: list[int],
) -> str:
    """Emit code to reconstruct one subclass output. Returns the variable name."""
    inner_dict_var = buf.fresh_name("_out_inner")
    entries: list[str] = []

    for attr, attr_meta in meta.attrs.items():
        match attr_meta:
            case PlainTensorMeta() | OpaqueMeta():
                idx = out_idx_ref[0]
                out_idx_ref[0] += 1
                entries.append(f"{attr!r}: unwrapped_outs[{idx}]")
            case SubclassCreationMeta():
                nested_var = _codegen_wrap_subclass(buf, attr_meta, out_idx_ref)
                entries.append(f"{attr!r}: {nested_var}")

    buf.writeline(f"{inner_dict_var} = {{{', '.join(entries)}}}")

    # Reconstruct outer_size and outer_stride
    size_placeholders = _compute_placeholders(meta.outer_size)
    stride_placeholders = _compute_placeholders(meta.outer_stride)

    def _build_tuple(
        outer: Iterable[None | int | SymInt], placeholders: list[bool]
    ) -> str:
        parts: list[str] = []
        for val, is_sym in zip(outer, placeholders):
            if is_sym:
                idx = out_idx_ref[0]
                out_idx_ref[0] += 1
                parts.append(f"unwrapped_outs[{idx}]")
            else:
                parts.append(repr(_concrete_value(val)))
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    size_expr = _build_tuple(meta.outer_size, size_placeholders)
    stride_expr = _build_tuple(meta.outer_stride, stride_placeholders)

    type_name = buf.bind_value(
        "_subclass_type",
        meta.original_subclass_type or type(meta.original_subclass),
    )
    meta_name = buf.bind_value("_meta", meta.meta)

    result_var = buf.fresh_name("_out")
    buf.writeline(
        f"{result_var} = {type_name}.__tensor_unflatten__("
        f"{inner_dict_var}, {meta_name}, {size_expr}, {stride_expr})"
    )
    return result_var


def _emit_output_wrapping(
    buf: PySourceBuilder,
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
) -> tuple[list[str], int]:
    """Emit wrapping code for output metas.

    Returns (result_exprs, num_args_tallied) where result_exprs are Python
    expression strings referencing each wrapped output.
    """
    out_idx_ref = [0]
    result_exprs: list[str] = []
    num_args_tallied = 0

    for meta in out_metas:
        if isinstance(meta, PlainTensorMeta):
            result_exprs.append(f"unwrapped_outs[{meta.unwrapped_idx}]")
            num_args_tallied += 1
            out_idx_ref[0] = max(out_idx_ref[0], meta.unwrapped_idx + 1)
        else:
            result_var = _codegen_wrap_subclass(buf, meta, out_idx_ref)
            result_exprs.append(result_var)
            num_args_tallied += meta.arg_count

    return result_exprs, num_args_tallied


def _emit_input_unwrapping(
    buf: PySourceBuilder,
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    frozen_inp_indices: frozenset[int] = frozenset(),
    include_symints: bool = True,
) -> None:
    """Emit unwrapping code for input metas into unwrapped_args.

    Caller must have already emitted ``unwrapped_args = []``.
    """
    for i, meta in enumerate(inp_metas):
        if isinstance(meta, PlainTensorMeta):
            buf.writeline(f"unwrapped_args.append(args[{i}])")
        elif i in frozen_inp_indices:
            # Frozen by inductor freezing: constant already baked into graph.
            buf.writeline("unwrapped_args.append(None)")
        else:
            inp_var = buf.fresh_name("_inp")
            type_name = buf.bind_value(
                "_expected_type",
                meta.original_subclass_type or type(meta.original_subclass),
            )
            buf.writeline(f"{inp_var} = args[{i}]")
            buf.writeline(
                f"assert type({inp_var}) is {type_name}, "
                f"f'expected {{{type_name}}}, got {{type({inp_var})}}'",
            )
            _codegen_unwrap_subclass(
                buf, meta, inp_var, include_symints=include_symints
            )


def _codegen_subclass_wrapper_source(
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_indices: list[int] | None = None,
) -> tuple[str, dict[str, object]]:
    """Generate source and globals for a subclass wrapper.

    Returns (source, globals_dict).  The globals_dict will NOT contain
    ``compiled_fn`` — the caller is responsible for adding it before exec.
    """
    buf = PySourceBuilder("inner_fn", args="args")

    with buf.indent():
        # --- Resolve AsyncCollectiveTensors ---
        # ACTs are transient eager-mode wrappers for async collective overlap.
        # Inductor triton kernels bypass __torch_dispatch__, so we must call
        # trigger_wait() before the compiled graph uses the data.
        if act_input_indices:
            for i in act_input_indices:
                buf.writeline(f"args[{i}] = args[{i}].trigger_wait()")

        # --- Input unwrapping ---
        buf.writeline("unwrapped_args = []")
        _emit_input_unwrapping(buf, inp_metas, frozen_inp_indices=frozen_inp_indices)

        # Pass through any trailing args not covered by inp_metas
        # (e.g. rng seed/offset added by FunctionalizedRngRuntimeWrapper).
        num_inp_metas = len(inp_metas)
        buf.writeline(f"unwrapped_args.extend(args[{num_inp_metas}:])")
        buf.writeline("args.clear()")

        # --- Call compiled function ---
        buf.writeline("unwrapped_outs = compiled_fn(unwrapped_args)")

        # --- Output wrapping ---
        result_exprs, num_args_tallied = _emit_output_wrapping(buf, out_metas)
        result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
        if num_fw_outs_saved_for_bw is not None:
            buf.writeline(
                f"return {result_tuple} + tuple(unwrapped_outs[{num_args_tallied}:])"
            )
        else:
            buf.writeline(f"return {result_tuple}")

    return buf.getvalue(), buf.globals


def _codegen_subclass_wrap_source(
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
) -> tuple[str, dict[str, object]]:
    """Generate source for wrapping flat outputs into subclasses.

    Used for the backward epilogue. Shares output-wrapping logic with
    _codegen_subclass_wrapper_source via _emit_output_wrapping.
    """
    buf = PySourceBuilder("wrap_fn", args="unwrapped_outs")
    with buf.indent():
        result_exprs, _ = _emit_output_wrapping(buf, out_metas)
        result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
        buf.writeline(f"return {result_tuple}")
    return buf.getvalue(), buf.globals


def _compile_and_exec_source(
    source: str,
    globals_dict: dict[str, object],
    fn_name: str,
    artifact_name: str,
    wrapped_fn: Callable[..., object] | None = None,
) -> Callable[..., object]:
    """Compile generated source, exec it, and return the named function.

    If wrapped_fn is provided, applies functools.update_wrapper so that
    __wrapped__ and __dict__ (e.g. _fx_graph_cache_key) propagate to the
    generated function.
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Generated %s:\n%s", artifact_name, source)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: source,
    )

    # Use a path under torch/_functorch/ so the code object is recognized by
    # dynamo's MOD_SKIPLIST. The eval frame hook stays active during the entire
    # torch.compile(fn)(*args) call (to handle graph breaks and resume functions),
    # so codegen'd functions called during backward get intercepted even though
    # no tracing is active. A real path makes them skip automatically.
    code = compile(source, f"{__file__}:codegen({artifact_name})", "exec")
    local_dict: dict[str, object] = {}
    exec(code, globals_dict, local_dict)
    fn = local_dict[fn_name]
    if wrapped_fn is not None:
        functools.update_wrapper(fn, wrapped_fn)  # type: ignore[arg-type]
    return fn  # type: ignore[return-value]


def codegen_backward_subclass_fns(
    grad_input_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None,
) -> tuple[Callable[..., object], Callable[..., object] | None]:
    """Generate codegen'd unwrap and wrap functions for the backward pass.

    Returns (unwrap_fn, wrap_fn). unwrap_fn is used by the backward prologue
    to unwrap non-tangent subclass inputs (always an identity in AOT dispatch
    since the compiled forward operates on unwrapped inner tensors). wrap_fn
    is used by the backward epilogue to wrap flat grad inputs back into
    subclasses; it is None when grad_input_metas is None.
    """
    source = "def unwrap_fn(args):\n    return list(args)"
    globals_dict: dict[str, object] = {}
    unwrap_fn = _compile_and_exec_source(
        source, globals_dict, "unwrap_fn", "backward_subclass_unwrap"
    )

    wrap_fn = None
    if grad_input_metas is not None:
        wrap_source, wrap_globals = _codegen_subclass_wrap_source(grad_input_metas)
        wrap_fn = _compile_and_exec_source(
            wrap_source, wrap_globals, "wrap_fn", "backward_subclass_wrapper"
        )

    return unwrap_fn, wrap_fn


def codegen_subclass_wrapper(
    compiled_fn: Callable[..., object],
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_indices: list[int] | None = None,
) -> Callable[..., object]:
    """Generate a specialized wrapper function for subclass unwrap/wrap."""
    source, globals_dict = _codegen_subclass_wrapper_source(
        inp_metas,
        out_metas,
        num_fw_outs_saved_for_bw,
        frozen_inp_indices,
        act_input_indices=act_input_indices,
    )
    globals_dict["compiled_fn"] = compiled_fn
    return _compile_and_exec_source(
        source, globals_dict, "inner_fn", "subclass_wrapper", wrapped_fn=compiled_fn
    )
