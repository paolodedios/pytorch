# Owner(s): ["oncall: pt2"]
import collections
import dataclasses
import enum
import functools
import math

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.subclass_codegen import GeneratedSource
from torch._functorch._aot_autograd.to_standalone_python import (
    _compose_standalone_module,
    _emit_importable,
    _emit_value,
    _find_effectful_op,
    _REBUILD_HELPER,
)
from torch._functorch.aot_autograd import compile_to_python
from torch._higher_order_ops.effects import _get_effect
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase


def _capture(m, x, tracing_mode="real"):
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), the same shape ``torch.compiler.precompile`` feeds the AOT lowering. The
    flat-input ordering returned by ``_flat_inputs`` MUST match this order."""
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        return pytree.tree_flatten(out)[0]

    with torch.enable_grad():
        return make_fx(flat_fn, tracing_mode=tracing_mode)(pb + [x])


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


def _emit(obj):
    """Emit source for ``obj`` and return ``(expr, sorted_imports)``."""
    imports: set[str] = set()
    return _emit_value(obj, imports), sorted(imports)


def _roundtrip(obj):
    """Emit ``obj`` as source, then exec the imports + ``_rebuild`` helper and eval the
    expression -- the same way the generated standalone module reconstructs it."""
    expr, imports = _emit(obj)
    ns: dict = {}
    if "_rebuild(" in expr:
        exec("\n".join(_REBUILD_HELPER), ns)
    for stmt in imports:
        exec(stmt, ns)
    return eval(expr, ns)


# Module-level fixtures: _emit_importable rejects any object whose __qualname__ carries a
# "<locals>" component, so enums / dataclasses / namedtuples used in the emission tests
# must be defined at module scope to be source-reconstructible.
class _Color(enum.Enum):
    RED = 1
    BLUE = 2


class _IColor(enum.IntEnum):
    RED = 1
    BLUE = 2


_Point = collections.namedtuple("_Point", ["x", "y"])


@dataclasses.dataclass
class _PlainDC:
    a: int
    b: str


@dataclasses.dataclass
class _DerivedDC:
    a: int
    derived: int = dataclasses.field(init=False, default=0)

    def __post_init__(self):
        # Re-derived from an init field, so a constructor call reproduces it: this still
        # round-trips and must NOT be rejected.
        self.derived = self.a * 2


@dataclasses.dataclass
class _StatefulDC:
    a: int
    extra: int = dataclasses.field(init=False, default=0)


class _MyInt(int):
    def __repr__(self):
        return f"_MyInt({int(self)})"


class _ReducesToLoadFromBytes:
    # A non-Tensor wrapper whose reduce delegates to storage bytes -- emitting it as
    # source would embed raw bytes and require a pickle.loads-equivalent at exec time.
    def __reduce_ex__(self, protocol):
        return (torch.storage._load_from_bytes, (b"\x00\x01\x02",))


def _ss_new(cls):
    return cls.__new__(cls)


class _StateSetterObj:
    # __reduce_ex__ returns the protocol-5 6-tuple form whose last element is a
    # state_setter; _rebuild cannot apply it, so emission must reject this object.
    def __reduce_ex__(self, protocol):
        return (_ss_new, (type(self),), {"x": 1}, None, None, lambda obj, state: None)


def _make_holder(value):
    return _Holder(value)


class _Holder:
    # Reduces to a top-level factory call (the generic-callable branch of
    # _emit_via_reduce: func is neither copyreg.__newobj__ nor __newobj_ex__).
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return (_make_holder, (self.value,))

    def __eq__(self, other):
        return isinstance(other, _Holder) and self.value == other.value


class _NewObjEx:
    # __getnewargs_ex__ makes __reduce_ex__(2) use copyreg.__newobj_ex__ with non-empty
    # kwargs, exercising that branch of _emit_via_reduce.
    def __new__(cls, a, b):
        obj = object.__new__(cls)
        obj.a = a
        obj.b = b
        return obj

    def __getnewargs_ex__(self):
        return ((self.a,), {"b": self.b})

    def __eq__(self, other):
        return isinstance(other, _NewObjEx) and self.a == other.a and self.b == other.b


class _Pointwise(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2.0 + 1.0)


class _ViewAlias(torch.nn.Module):
    def forward(self, x):
        return x.view(-1)


class _SumDim1(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


class _BufferMutate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x):
        self.b.add_(x.sum())
        return x + self.b


class TestAOTCompileToPython(TestCase):
    # End-to-end coverage of the functorch composition layer: compile_to_python composes
    # AOTAutograd's codegen'd runtime wrappers (prelude/epilogue) around the inner Inductor
    # call into one standalone module, and the emitted module must match eager. All CPU.
    def _compose(self, m, x):
        gm = _capture(m, x)
        src, cache = compile_to_python(
            gm, _flat_inputs(m, x), options={"graph_partition": False}
        )
        return src, cache

    def _assert_composed(self, src):
        # Structural markers proving this is the COMPOSED module (not just the inner
        # inductor output): the outer entry takes flat_inputs, the inner call is captured
        # as _inner_call, and each wrapper is re-exec'd via _exec_wrapper / _orchestration.
        self.assertIn("def call(flat_inputs):", src)
        self.assertIn("_inner_call = call", src)
        self.assertIn("def _exec_wrapper(", src)
        self.assertIn("_orchestration", src)
        # Auditability guarantee: no pickle.loads / base64 blob in the emitted module.
        self.assertNotIn("pickle.loads", src)

    def test_pointwise_runs_like_eager(self):
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        src, cache = self._compose(m, x)
        self._assert_composed(src)
        # Return contract: cache is the opaque acceleration bytes or None.
        self.assertIsInstance(cache, (bytes, type(None)))
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_linear_addmm_runs_like_eager(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = self._compose(m, x)
        self._assert_composed(src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_sequential_linear_relu_runs_like_eager(self):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        src, _cache = self._compose(m, x)
        self._assert_composed(src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_reduction_runs_like_eager(self):
        m = _SumDim1().eval()
        x = torch.randn(6, 7)
        src, _cache = self._compose(m, x)
        self._assert_composed(src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_dynamic_shapes_runs_at_multiple_shapes(self):
        # dynamic_shapes="from_graph" on a symbolically-traced graph composes one module
        # keyed on symbolic sizes rather than baked constants, and that single module runs
        # at multiple shapes. (The default "from_example_inputs" specializes instead.)
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x, tracing_mode="symbolic")
        src, _cache = compile_to_python(
            gm,
            _flat_inputs(m, x),
            dynamic_shapes="from_graph",
            options={"graph_partition": False},
        )
        self._assert_composed(src)
        fn = _exec(src)
        for n in (8, 16, 5):
            xi = torch.randn(n, 4)
            with torch.no_grad():
                self.assertEqual(fn(_flat_inputs(m, xi))[0], m(xi))

    def test_input_mutation_copy_back_runs_like_eager(self):
        # A buffer mutated in place exercises AOTAutograd's mutation epilogue (input copy-
        # back): the composed call must reflect the mutation onto the passed-in buffer
        # tensor, exactly as eager mutates m.b. Compare both the output AND the mutated
        # input.
        m = _BufferMutate().eval()
        x = torch.randn(4)
        src, _cache = self._compose(m, x)
        self._assert_composed(src)

        eager = _BufferMutate().eval()
        eager_out = eager(x)

        buf = torch.zeros(4)
        with torch.no_grad():
            composed_out = _exec(src)([buf, x])[0]
        self.assertEqual(composed_out, eager_out)
        self.assertEqual(buf, eager.b)

    def test_output_alias_regen_runs_like_eager(self):
        # An output that aliases an input exercises AOTAutograd's output-alias regeneration
        # (the _alias_fn / gen_alias_from_base path, wired into the orchestration via a
        # forwarding shim that _resolve_global must unwrap). The composed output must both
        # equal eager AND alias the input's storage, exactly as eager's view does.
        m = _ViewAlias().eval()
        x = torch.randn(4, 4)
        src, _cache = self._compose(m, x)
        self._assert_composed(src)
        self.assertIn("gen_alias_from_base", src)
        xc = x.clone()
        with torch.no_grad():
            out = _exec(src)([xc])[0]
        self.assertEqual(out, m(x))
        self.assertEqual(
            out.untyped_storage().data_ptr(), xc.untyped_storage().data_ptr()
        )

    def test_tensor_subclass_wrap_unwrap_runs_like_eager(self):
        # The headline feature: a tensor-subclass input exercises AOTAutograd's subclass
        # flatten/unflatten wrapper plus baked subclass metadata. The composed module must
        # unwrap the subclass for the inner dense call and re-wrap the output as the same
        # subclass, matching eager.
        from torch.testing._internal.two_tensor import TwoTensor

        def f(x):
            return x * 2.0 + 1.0

        tt = TwoTensor(torch.randn(4, 4), torch.randn(4, 4))
        gm = make_fx(f, tracing_mode="real")(tt)
        src, _cache = compile_to_python(gm, [tt], options={"graph_partition": False})
        self._assert_composed(src)
        with torch.no_grad():
            out = _exec(src)([tt])[0]
        eager = f(tt)
        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, eager.a)
        self.assertEqual(out.b, eager.b)

    def test_rejects_effectful_op(self):
        # A graph carrying an effectful op (here aten._print) is rejected up front with a
        # concrete NotImplementedError -- effect tokens thread through a calling convention
        # the standalone composition does not reproduce.
        g = fx.Graph()
        a = g.placeholder("a")
        g.call_function(torch.ops.aten._print.default, ("hello",))
        g.output((a,))
        gm = fx.GraphModule(torch.nn.Module(), g)
        with self.assertRaisesRegex(NotImplementedError, "effectful op"):
            compile_to_python(gm, [torch.randn(3)])

    def test_rejects_non_graphmodule(self):
        # The effectful-op scan dereferences gm.graph before reaching inductor's own check,
        # so the functorch layer must reject a non-GraphModule with a clean TypeError rather
        # than an opaque AttributeError.
        with self.assertRaisesRegex(TypeError, "expects a post-AOTAutograd"):
            compile_to_python("not a graph module", [])


class TestAOTCompileToPythonHelpers(TestCase):
    # Unit coverage of the source-emission helpers: every _emit_value branch round-trips
    # (or raises on a non-source-expressible leaf), so the standalone artifact stays
    # auditable and pickle.loads-free.
    def test_none_and_exact_builtin_scalars(self):
        self.assertEqual(_emit(None), ("None", []))
        self.assertEqual(_emit(True), ("True", []))
        self.assertEqual(_emit(42), ("42", []))
        self.assertEqual(_emit(3.5), ("3.5", []))
        self.assertEqual(_emit("ab"), ("'ab'", []))
        self.assertEqual(_emit(b"xy"), ("b'xy'", []))
        self.assertEqual(_emit(bytearray(b"xy")), ("bytearray(b'xy')", []))
        self.assertEqual(_emit(complex(1, 2)), ("(1+2j)", []))

    def test_non_finite_float_and_complex(self):
        self.assertEqual(_emit(float("inf")), ("float('inf')", []))
        self.assertEqual(_emit(float("-inf")), ("float('-inf')", []))
        self.assertEqual(_emit(float("nan")), ("float('nan')", []))
        self.assertTrue(math.isnan(_roundtrip(float("nan"))))
        self.assertEqual(
            _emit(complex(float("inf"), 0.0))[0], "complex(float('inf'), 0.0)"
        )
        rt = _roundtrip(complex(float("inf"), 0.0))
        self.assertTrue(math.isinf(rt.real) and rt.imag == 0.0)

    def test_int_subclass_falls_through_repr(self):
        # An int subclass with a constructor-style __repr__ must NOT take the exact-type
        # repr branch (which would emit a NameError / lose its type); it round-trips via
        # the reduce path and keeps its type.
        rt = _roundtrip(_MyInt(5))
        self.assertEqual(rt, 5)
        self.assertIs(type(rt), _MyInt)

    def test_torch_scalar_singletons(self):
        self.assertEqual(_emit(torch.float32), ("torch.float32", ["import torch"]))
        self.assertEqual(_emit(torch.strided), ("torch.strided", ["import torch"]))
        self.assertEqual(
            _emit(torch.contiguous_format),
            ("torch.contiguous_format", ["import torch"]),
        )
        self.assertIs(_roundtrip(torch.float32), torch.float32)

    def test_torch_device_size(self):
        self.assertEqual(
            _emit(torch.device("cpu")), ("torch.device('cpu')", ["import torch"])
        )
        self.assertEqual(_roundtrip(torch.device("cpu")), torch.device("cpu"))
        self.assertEqual(
            _emit(torch.Size([2, 3])), ("torch.Size([2, 3])", ["import torch"])
        )
        self.assertEqual(_roundtrip(torch.Size([2, 3])), torch.Size([2, 3]))

    def test_importable_class_function_module(self):
        self.assertIs(_roundtrip(torch.nn.Linear), torch.nn.Linear)
        self.assertIs(_roundtrip(math), math)
        self.assertEqual(_emit(math), ("math", ["import math"]))

    def test_importable_rejects_lambda_and_local(self):
        with self.assertRaises(NotImplementedError):
            _emit(lambda x: x)

        def _local():
            pass

        with self.assertRaises(NotImplementedError):
            _emit(_local)

    def test_enums(self):
        self.assertIs(_roundtrip(_Color.RED), _Color.RED)
        self.assertIs(_roundtrip(_Color.BLUE), _Color.BLUE)
        # IntEnum: repr is "<_IColor.RED: 1>" (invalid source), so it must take the enum
        # branch, not the repr branch.
        self.assertIs(_roundtrip(_IColor.RED), _IColor.RED)

    def test_functools_partial(self):
        # _roundtrip rebuilds the partial object itself; invoking it then applies the
        # baked func/args/keywords.
        p = _roundtrip(functools.partial(_PlainDC, b="z"))
        self.assertEqual(p(a=1), _PlainDC(1, "z"))
        _expr, imports = _emit(functools.partial(int, "10", base=2))
        self.assertIn("import functools", imports)
        self.assertEqual(_roundtrip(functools.partial(int, "10", base=2))(), 2)

    def test_containers(self):
        self.assertEqual(_emit((1, 2)), ("(1, 2)", []))
        self.assertEqual(_emit((1,)), ("(1,)", []))
        self.assertEqual(_emit(()), ("()", []))
        self.assertEqual(_emit([1, 2]), ("[1, 2]", []))
        self.assertEqual(_emit([]), ("[]", []))
        self.assertEqual(_emit({"a": 1}), ("{'a': 1}", []))
        self.assertEqual(_roundtrip({"k": [1, 2, (3,)]}), {"k": [1, 2, (3,)]})

    def test_namedtuple(self):
        rt = _roundtrip(_Point(1, 2))
        self.assertEqual(rt, _Point(1, 2))
        self.assertIs(type(rt), _Point)

    def test_set_canonical_ordering_is_deterministic(self):
        # Set iteration order is not byte-stable across processes; emission sorts to a
        # canonical order so the artifact is reproducible.
        self.assertEqual(_emit({3, 1, 2})[0], "set([1, 2, 3])")
        self.assertEqual(_emit(frozenset({3, 1, 2}))[0], "frozenset([1, 2, 3])")
        self.assertEqual(_emit({3, 1, 2})[0], _emit({2, 3, 1})[0])
        # Unorderable elements (int vs str) fall back to sorting by emitted source.
        self.assertEqual(_emit({1, "a"})[0], "set(['a', 1])")
        self.assertEqual(_emit({1, "a", "b"})[0], _emit({"b", 1, "a"})[0])

    def test_plain_dataclass_round_trips(self):
        rt = _roundtrip(_PlainDC(1, "z"))
        self.assertEqual(rt, _PlainDC(1, "z"))

    def test_dataclass_with_derived_post_init_round_trips(self):
        rt = _roundtrip(_DerivedDC(5))
        self.assertEqual(rt, _DerivedDC(5))
        self.assertEqual(rt.derived, 10)

    def test_stateful_dataclass_is_rejected(self):
        # A non-init field mutated to a value the constructor cannot reproduce makes the
        # rebuilt instance compare unequal: emitting only the init fields would silently
        # drop that state, so it must raise.
        obj = _StatefulDC(5)
        obj.extra = 99
        with self.assertRaisesRegex(NotImplementedError, "does not round-trip"):
            _emit(obj)

    def test_live_tensor_and_storage_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "live Tensor"):
            _emit(torch.zeros(2))
        with self.assertRaisesRegex(NotImplementedError, "live UntypedStorage"):
            _emit(torch.zeros(3).untyped_storage())

    def test_reduce_to_load_from_bytes_rejected(self):
        # Regression: the previous guard compared the __reduce_ex__ METHOD against
        # _load_from_bytes and never fired; the callable is the reduce RESULT's func.
        # A wrapper whose reduce is _load_from_bytes must be rejected, not silently emit
        # raw bytes + a pickle.loads-equivalent.
        with self.assertRaisesRegex(NotImplementedError, "_load_from_bytes"):
            _emit(_ReducesToLoadFromBytes())

    def test_state_setter_reduce_rejected(self):
        # Regression: a protocol-5 6-tuple reduce carries a state_setter _rebuild cannot
        # apply; emitting _rebuild(base, state) would install state via the wrong
        # mechanism, so reject it.
        with self.assertRaisesRegex(NotImplementedError, "state_setter"):
            _emit(_StateSetterObj())

    def test_self_referential_container_rejected(self):
        a = [1, 2]
        a.append(a)
        with self.assertRaisesRegex(NotImplementedError, "self-referential"):
            _emit(a)
        d: dict = {}
        d["self"] = d
        with self.assertRaisesRegex(NotImplementedError, "self-referential"):
            _emit(d)

    def test_repeated_scalar_is_not_a_cycle(self):
        # Scalars are exempt from the identity cycle guard, so a list repeating a scalar
        # must still emit.
        self.assertEqual(_emit([0, 0, 0]), ("[0, 0, 0]", []))

    def test_emit_via_reduce_round_trips_opaque_object(self):
        # An opaque value object with a copyreg.__newobj__ reduce + dict state rebuilds via
        # cls.__new__ + _rebuild, emitted as source (no pickle.loads).
        try:
            from torch.distributed.tensor.placement_types import Shard
        except Exception:
            self.skipTest("DTensor placement_types unavailable")
        rt = _roundtrip(Shard(2))
        self.assertEqual(rt, Shard(2))

    def test_emit_via_reduce_generic_callable(self):
        # An object whose reduce is a plain top-level factory (not copyreg.__newobj__)
        # rebuilds via that factory call, emitted as source.
        rt = _roundtrip(_Holder(7))
        self.assertEqual(rt, _Holder(7))

    def test_emit_via_reduce_newobj_ex(self):
        # __getnewargs_ex__ drives the copyreg.__newobj_ex__ branch (cls.__new__ with
        # positional + keyword args) plus _rebuild for the dict state.
        rt = _roundtrip(_NewObjEx(1, b=2))
        self.assertEqual(rt, _NewObjEx(1, b=2))

    def test_emit_importable_rejects_non_round_tripping(self):
        # torch.add is a builtin whose __qualname__ does not round-trip via importlib.
        with self.assertRaises(NotImplementedError):
            _emit_importable(torch.add, set())

    def test_find_effectful_op_top_level(self):
        g = fx.Graph()
        a = g.placeholder("a")
        g.call_function(torch.ops.aten._print.default, ("hi",))
        g.output((a,))
        gm = fx.GraphModule(torch.nn.Module(), g)
        self.assertIs(
            _find_effectful_op(gm, _get_effect), torch.ops.aten._print.default
        )

    def test_find_effectful_op_nested_in_subgraph(self):
        # An effect nested inside a child GraphModule reached via get_attr must be found.
        child = fx.Graph()
        ca = child.placeholder("a")
        child.call_function(torch.ops.aten._print.default, ("hi",))
        child.output((ca,))
        child_gm = fx.GraphModule(torch.nn.Module(), child)

        parent = fx.Graph()
        pa = parent.placeholder("a")
        parent.get_attr("sub")
        parent.output((pa,))
        root = torch.nn.Module()
        root.sub = child_gm
        parent_gm = fx.GraphModule(root, parent)
        self.assertIs(
            _find_effectful_op(parent_gm, _get_effect), torch.ops.aten._print.default
        )

    def test_find_effectful_op_none_when_pure(self):
        m = _Pointwise().eval()
        gm = _capture(m, torch.randn(4, 4))
        self.assertIsNone(_find_effectful_op(gm, _get_effect))


class TestAOTComposeGuards(TestCase):
    # The composer's defensive guards (which reject rather than emit a subtly-wrong module)
    # only fire if AOTAutograd's codegen drifts, so drive them directly with hand-built
    # GeneratedSource objects rather than waiting for an upstream regression.
    _ORCH_SRC = (
        "def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):\n"
        "    return _compiled_fn_(args)\n"
    )
    _CHAIN_SRC = "def inner_fn(args):\n    return compiled_fn(args)\n"

    def test_orchestration_signature_guard(self):
        # The generated call invokes the orchestration positionally, so a changed signature
        # must fail loudly rather than silently pass wrong arguments.
        bad_orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            "def _runtime_wrapper(wrong, args):\n    return None\n",
            {},
            lambda: None,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "orchestration wrapper signature"
        ):
            _compose_standalone_module("def call(args):\n    return args\n", [bad_orch])

    def test_chain_head_order_inversion_guard(self):
        # Capture order is assumed innermost-to-outermost. Feed it OUTER-first (inverted):
        # the outer wrapper (wraps the inner wrapper) is captured before the inner wrapper
        # (wraps the dense call), so the "last with an inner-ref" head is actually wrapped.
        def inner_fn(args):
            return args

        def fn_a(args):
            return args

        def fn_b(args):
            return args

        def orch_fn():
            return None

        orch = GeneratedSource(
            "runtime_wrapper_orchestration",
            "_runtime_wrapper",
            self._ORCH_SRC,
            {},
            orch_fn,
        )
        # outer (fn_b) wraps the inner wrapper (fn_a); inner (fn_a) wraps the dense call.
        outer = GeneratedSource(
            "dedup_wrapper", "inner_fn", self._CHAIN_SRC, {"compiled_fn": fn_a}, fn_b
        )
        inner = GeneratedSource(
            "dedup_wrapper",
            "inner_fn",
            self._CHAIN_SRC,
            {"compiled_fn": inner_fn},
            fn_a,
        )
        with self.assertRaisesRegex(NotImplementedError, "innermost-to-outermost"):
            _compose_standalone_module(
                "def call(args):\n    return args\n", [outer, inner, orch]
            )


if __name__ == "__main__":
    run_tests()
