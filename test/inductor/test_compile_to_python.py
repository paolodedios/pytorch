# Owner(s): ["module: inductor"]
import unittest

import torch
import torch._functorch.aot_autograd as aot
import torch._inductor.config as inductor_config
import torch.utils._pytree as pytree
from torch._inductor import compile_to_python as inductor_compile_to_python
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


def _capture(m, x):
    """Capture m(x) as a flat-input ATen graph (params+buffers then x lifted to
    inputs), mirroring how torch.compiler.precompile feeds standalone graphs to a backend.

    NOTE: tracing runs m(x) once, which mutates m's buffers for stateful modules;
    callers should capture from a throwaway model and run on a fresh one.
    """
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
        leaves, _ = pytree.tree_flatten(out)
        return leaves

    with torch.enable_grad():
        gm = make_fx(flat_fn)(pb + [x])
    return gm


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


class TestInductorCompileToPython(TestCase):
    # torch._inductor.compile_to_python returns the INNER call only (no epilogue);
    # for a dense graph that is the whole computation, run under no_grad.
    def test_inner_call_dense_matches_eager(self):
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)

        inner_src, cache = inductor_compile_to_python(gm, _flat_inputs(m, x))
        self.assertIsInstance(inner_src, str)
        self.assertIsNotNone(cache)  # non-mutating graph is serializable

        call = _exec(inner_src)
        with torch.no_grad():
            out = call(_flat_inputs(m, x))
        self.assertEqual(out[0], m(x))

    def test_concurrent_compile_restores_save_output_code_hook(self):
        # compile_to_python installs a process-global GraphLowering.save_output_code hook
        # to capture codegen, serialized by an internal lock and restored in a finally.
        # Run N threads each lowering a DISTINCT graph (distinct output shape so they do
        # not cache-collide); each result must be runnable and match eager, AND the hook
        # must be restored to its ambient value (None here) afterward -- a regression for
        # the lock fix (a leaked/clobbered hook would corrupt a concurrent caller).
        import threading

        from torch._inductor.graph import GraphLowering

        self.assertIsNone(GraphLowering.save_output_code)
        n = 4

        # Capture all graphs SERIALLY first: make_fx is not thread-safe (it mutates a
        # process-global tracer patcher); only the lock-protected lowering runs concurrently.
        def make_spec(i):
            torch.manual_seed(i)
            m = torch.nn.Linear(4, 3 + i).eval()  # distinct output shape per thread
            x = torch.randn(2, 4)
            return _capture(m, x), _flat_inputs(m, x), m, x

        specs = [make_spec(i) for i in range(n)]
        results: dict[int, bool] = {}
        errors: dict[int, str] = {}

        def worker(i):
            try:
                gm, inps, m, x = specs[i]
                inner_src, _cache = inductor_compile_to_python(gm, inps)
                call = _exec(inner_src)
                with torch.no_grad():
                    out = call(inps)
                results[i] = torch.allclose(out[0], m(x))
            except Exception as e:
                errors[i] = f"{type(e).__name__}: {e}"

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, {})
        self.assertEqual(results, dict.fromkeys(range(n), True))
        # The hook is restored to its ambient (None) value, not leaked from any thread.
        self.assertIsNone(GraphLowering.save_output_code)

    def test_foreign_thread_codegen_does_not_pollute_capture(self):
        # compile_to_python's capture_hook lives on the PROCESS-GLOBAL
        # GraphLowering.save_output_code, so a codegen happening on ANOTHER thread
        # (e.g. a concurrent normal torch.compile) fires the same hook. The hook has an
        # owner-thread guard (it returns immediately unless the firing thread is the one
        # that installed it); without that guard the foreign source gets appended to the
        # capture list and _extract_runnable_module raises "found 2". This reproduces the
        # race deterministically: a foreign daemon thread fires whatever hook is currently
        # installed, with V.graph set to a stand-in graph whose name does NOT start with
        # 'benchmark_' (so the bug-version would append), hammering continuously while the
        # main thread's capture window is open -- so the foreign call lands DURING it.
        import threading
        import types

        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V

        self.assertIsNone(GraphLowering.save_output_code)

        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        inps = _flat_inputs(m, x)

        stop = threading.Event()
        fire_count = 0
        foreign_src = "def call(args):\n    return args\n"
        foreign_graph = types.SimpleNamespace(name="foreign_graph")

        def foreign():
            nonlocal fire_count
            while not stop.is_set():
                hook = GraphLowering.save_output_code
                if hook is not None:
                    # Mirror a real foreign codegen: V.graph is the GraphLowering being
                    # lowered when the hook fires. With the guard, capture_hook returns on
                    # the thread check BEFORE reading V.graph, so this never pollutes; without
                    # it, capture_hook reads V.graph.name (not 'benchmark_*') and appends.
                    with V.set_graph_handler(foreign_graph):
                        hook(foreign_src)
                    fire_count += 1

        t = threading.Thread(target=foreign, daemon=True)
        t.start()
        try:
            inner_src, _cache = inductor_compile_to_python(gm, inps)
        finally:
            stop.set()
            t.join(timeout=10)

        # The foreign thread must have actually fired the installed hook during the window
        # (otherwise the test is vacuous: it would pass even with the guard removed).
        self.assertGreater(fire_count, 0)

        # The returned source runs standalone and matches eager (no spurious "found 2").
        call = _exec(inner_src)
        with torch.no_grad():
            out = call(inps)
        self.assertEqual(out[0], m(x))

        # The hook is restored to its ambient (None) value afterward.
        self.assertIsNone(GraphLowering.save_output_code)


class TestAotCompileToPython(TestCase):
    # torch._functorch.aot_autograd.compile_to_python returns the full self-contained
    # module (inner call + AOTAutograd's codegen'd epilogue).
    def test_dense_matches_eager(self):
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)

        src, cache = aot.compile_to_python(gm, _flat_inputs(m, x))
        self.assertIsInstance(src, str)
        self.assertIsNotNone(cache)
        # The epilogue is AOTAutograd's own codegen, not a hand-rolled driver.
        self.assertIn("_runtime_wrapper", src)

        call = _exec(src)
        self.assertEqual(call(_flat_inputs(m, x))[0], m(x))

    def test_self_contained_runs_without_cache(self):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        src, _cache = aot.compile_to_python(gm, _flat_inputs(m, x))
        call = _exec(src)
        self.assertEqual(call(_flat_inputs(m, x))[0], m(x))

    def test_no_benchmark_harness_in_output(self):
        # The export artifact is meant to run, not to be profiled, so the Inductor
        # benchmark/debug harness (get_args / benchmark_compiled_module / __main__)
        # and the no-op compile-time auto-tuning docstring are suppressed at codegen
        # time (not stripped afterward) -- unlike the default output code tlparse uses.
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        src, _cache = aot.compile_to_python(gm, _flat_inputs(m, x))
        for marker in (
            "benchmark_compiled_module",
            "def get_args(",
            "compiled_module_main",
            "print_performance",
            # The compile-time auto-tuning docstring is also debug-only (GPU/triton).
            "Compile-time auto-tuning block",
        ):
            self.assertNotIn(marker, src)
        call = _exec(src)
        self.assertEqual(call(_flat_inputs(m, x))[0], m(x))

    def test_buffer_mutation_is_reflected(self):
        # BatchNorm in training mutates running stats. The composed module reflects
        # that onto the passed-in buffers via AOTAutograd's captured orchestration,
        # and matches eager.
        def fresh():
            torch.manual_seed(0)
            mm = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            mm.train()
            return mm

        x = torch.randn(8, 4)
        gm = _capture(fresh(), x)  # throwaway model mutated during tracing
        src, _cache = aot.compile_to_python(gm, _flat_inputs(fresh(), x))

        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()
        ref_rv = ref[1].running_var.clone()
        ref_nbt = ref[1].num_batches_tracked.clone()

        run = fresh()
        call = _exec(src)
        out = call(_flat_inputs(run, x))

        self.assertEqual(out[0], ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)
        self.assertEqual(run[1].running_var, ref_rv)
        self.assertEqual(run[1].num_batches_tracked, ref_nbt)
        self.assertNotEqual(float(run[1].running_mean.abs().sum()), 0.0)

    def test_duplicate_input(self):
        # The same tensor passed as two graph inputs goes through AOTAutograd's
        # dedup wrapper; the general composition handles it.
        t = torch.randn(4)

        def flat_fn(flat):
            return pytree.tree_flatten(flat[0] + flat[1])[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn)([t, t])
        src, _cache = aot.compile_to_python(gm, [t, t])
        call = _exec(src)
        self.assertEqual(call([t, t])[0], t + t)

    def test_functionalized_rng(self):
        # Functionalized RNG (dropout) threads seed/offset through the calling
        # convention; the RNG wrapper composes in (the verification is structural,
        # since the two RNG paths draw different masks).
        import torch._functorch.config as functorch_config

        x = torch.randn(64)

        def flat_fn(flat):
            return pytree.tree_flatten(
                torch.nn.functional.dropout(flat[0], 0.5, training=True)
            )[0]

        with functorch_config.patch(functionalize_rng_ops=True):
            with torch.enable_grad():
                gm = make_fx(flat_fn)([x])
            src, _cache = aot.compile_to_python(gm, [x])
            self.assertIn("CUDARngStateHelper", src)
            call = _exec(src)
            out = call([x])
        self.assertEqual(out[0].shape, x.shape)
        # Dropout zeros some elements: a valid masked result, not the identity.
        self.assertTrue((out[0] == 0).any())

    def test_effectful_op_not_supported(self):
        # An effectful custom op makes the Inductor artifact non-saveable, so the
        # inner code cannot be extracted to standalone source -> clean error.
        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
        from torch.library import _scoped_library

        with _scoped_library("mlctp", "FRAGMENT") as lib:
            lib.define("eff(Tensor x) -> Tensor")
            lib.impl("eff", lambda x: x + 1.0, "CompositeExplicitAutograd")
            lib.impl("eff", lambda x: torch.empty_like(x), "Meta")
            op = torch.ops.mlctp.eff.default
            _register_effectful_op(op, _EffectType.ORDERED)
            try:
                x = torch.randn(4)

                def flat_fn(flat):
                    return pytree.tree_flatten(torch.ops.mlctp.eff(flat[0]))[0]

                with torch.enable_grad():
                    gm = make_fx(flat_fn)([x])
                with self.assertRaisesRegex(
                    NotImplementedError, "cannot lower this graph to standalone source"
                ):
                    aot.compile_to_python(gm, [x])
            finally:
                _register_effectful_op(op, None)

    def test_effectful_op_nested_in_hop_not_supported(self):
        # An effectful op nested inside a HOP subgraph (a torch.cond branch) must also be
        # rejected: the effect scan recurses into subgraphs, not just top-level nodes.
        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
        from torch.library import _scoped_library

        with _scoped_library("mlctp", "FRAGMENT") as lib:
            lib.define("eff2(Tensor x) -> Tensor")
            lib.impl("eff2", lambda x: x + 1.0, "CompositeExplicitAutograd")
            lib.impl("eff2", lambda x: torch.empty_like(x), "Meta")
            op = torch.ops.mlctp.eff2.default
            _register_effectful_op(op, _EffectType.ORDERED)
            try:
                pred = torch.tensor(True)
                x = torch.randn(4)

                def flat_fn(flat):
                    out = torch.cond(
                        flat[0],
                        lambda a: torch.ops.mlctp.eff2(a),
                        lambda a: a + 1.0,
                        (flat[1],),
                    )
                    return pytree.tree_flatten(out)[0]

                with torch.enable_grad():
                    gm = make_fx(flat_fn)([pred, x])
                with self.assertRaisesRegex(
                    NotImplementedError, "cannot lower this graph to standalone source"
                ):
                    aot.compile_to_python(gm, [pred, x])
            finally:
                _register_effectful_op(op, None)

    def test_output_alias(self):
        # An output that is a view of an input goes through AOTAutograd's output-
        # alias epilogue (gen_alias_from_base); the composed module reproduces it.
        x = torch.randn(3, 4)

        def fn(a):
            return a.t()

        def flat_fn(flat):
            return pytree.tree_flatten(fn(*flat))[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn)([x])
        src, _cache = aot.compile_to_python(gm, [x])
        self.assertIn("gen_alias_from_base", src)
        # Runtime helpers are imported from the single stable runtime surface,
        # not scattered AOTAutograd internals.
        self.assertIn("standalone_runtime import gen_alias_from_base", src)
        # The view-replay recipe is reconstructed as plain source (ViewMetaSequence
        # via its factory), not embedded as a pickle blob.
        self.assertIn("ViewMetaSequence._from_parts", src)
        self.assertNotIn("_unpickle(", src)
        self.assertNotIn("import pickle", src)

        call = _exec(src)
        out = call([x])
        self.assertEqual(out[0], x.t())

    def test_dtensor_subclass(self):
        # DTensor (tensor-subclass) graph inputs/outputs go through AOTAutograd's
        # subclass wrap/unwrap; the composed module reproduces it. The output
        # subclass's flatten metadata (placements/spec) is reconstructed as plain
        # source -- including the placement objects via _rebuild -- not pickled.
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")
        import os

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        saved = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            for name, p in list(m.named_parameters()):
                setattr(
                    m,
                    name,
                    torch.nn.Parameter(
                        distribute_tensor(p.detach(), mesh, [Replicate()])
                    ),
                )
            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            ref = m(x)

            gm = _capture(m, x)
            src, _cache = aot.compile_to_python(gm, _flat_inputs(m, x))
            self.assertIn("__tensor_unflatten__", src)
            # Placement metadata is emitted as readable source, not a pickle blob,
            # so exec'ing the module never invokes pickle.loads.
            self.assertIn("placement_types.Replicate", src)
            self.assertNotIn("_unpickle(", src)
            self.assertNotIn("import pickle", src)

            call = _exec(src)
            out = call(_flat_inputs(m, x))
            self.assertEqual(out[0].to_local(), ref.to_local())
        finally:
            dist.destroy_process_group()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @unittest.skipUnless(TEST_CUDA, "max_autotune Triton benchmarking needs CUDA")
    def test_max_autotune_matmul_compiles(self):
        # A matmul lowered under max_autotune triggers Inductor's autotuning, which
        # benchmarks candidate kernels at compile time. compile_to_python must compose
        # through that nested-benchmark path (the fix) and produce a runnable module that
        # matches eager.
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")

        def flat_fn(flat):
            return pytree.tree_flatten(flat[0] @ flat[1])[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn)([a, b])
        with inductor_config.patch(max_autotune=True):
            src, _cache = aot.compile_to_python(gm, [a, b])
        call = _exec(src)
        out = call([a, b])
        self.assertEqual(out[0], a @ b)

    @unittest.skipUnless(TEST_CUDA, "max_autotune Triton benchmarking needs CUDA")
    def test_max_autotune_matmul_precompile(self):
        # The same nested-benchmark path, end-to-end through torch.compiler.precompile:
        # a matmul precompiled under max_autotune produces a working artifact.
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        with inductor_config.patch(max_autotune=True):
            code, cache = torch.compiler.precompile(lambda x, y: x @ y, a, b)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(a, b), a @ b)

    @unittest.skipUnless(TEST_CUDA, "max_autotune Triton benchmarking needs CUDA")
    def test_options_applied_as_inductor_config(self):
        # ``options`` is an inductor config-override dict folded into the config.patch the
        # lowering already enters (NOT forwarded as compile_fx kwargs). Passing
        # options={"max_autotune": True} must take effect as config -- driving the same
        # nested-benchmark autotuning path -- and still produce a runnable module that
        # matches eager.
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")

        def flat_fn(flat):
            return pytree.tree_flatten(flat[0] @ flat[1])[0]

        with torch.enable_grad():
            gm = make_fx(flat_fn)([a, b])
        src, _cache = aot.compile_to_python(gm, [a, b], options={"max_autotune": True})
        call = _exec(src)
        out = call([a, b])
        self.assertEqual(out[0], a @ b)

    def test_options_cannot_override_capture_critical_pin(self):
        # The two capture-critical pins (benchmark_harness, cpp_wrapper) always win over a
        # conflicting user ``options`` value, since the source-capture contract depends on
        # them. Asking for options={"benchmark_harness": True} must NOT leak the Inductor
        # benchmark/debug harness into the emitted source (the pin overrides it), and the
        # module must still run and match eager. CPU-runnable (no autotuning involved).
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        src, _cache = aot.compile_to_python(
            gm, _flat_inputs(m, x), options={"benchmark_harness": True}
        )
        for marker in (
            "benchmark_compiled_module",
            "def get_args(",
            "compiled_module_main",
            "print_performance",
        ):
            self.assertNotIn(marker, src)
        call = _exec(src)
        self.assertEqual(call(_flat_inputs(m, x))[0], m(x))

    def test_options_cpp_wrapper_pin_keeps_python_module(self):
        # The most consequential capture-critical pin, exercised via the ``options`` path:
        # cpp_wrapper would make Inductor emit a C++ ``call`` (no module-level python
        # ``def call(args)`` / ``call = runner.call`` to inline), so a python artifact
        # could not be extracted. compile_to_python pins cpp_wrapper OFF and the pin must
        # override a conflicting user option={"cpp_wrapper": True}: the emitted source must
        # still be a PYTHON module (no C++ ``extern "C"`` / CppWrapper marker, a module-
        # level call binding present) and run correctly. CPU-runnable, no autotuning.
        from torch._inductor.standalone_compile import _defines_module_level_call

        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        src, _cache = aot.compile_to_python(
            gm, _flat_inputs(m, x), options={"cpp_wrapper": True}
        )
        self.assertNotIn('extern "C"', src)  # not a C++ wrapper module
        self.assertNotIn("CppWrapperCpu", src)
        self.assertTrue(_defines_module_level_call(src))  # python ``call`` is present
        call = _exec(src)
        self.assertEqual(call(_flat_inputs(m, x))[0], m(x))

    def test_no_runnable_module_surfaces_clear_error(self):
        # A graph with no compute to lower (it returns its input unchanged, e.g.
        # ``x`` or ``x.detach()``) yields no module-level ``call`` -- the inductor
        # layer raises NoRunnableInductorModuleError, which torch.compiler.precompile
        # converts to a clear PrecompileError pointing at the "no compute" cause rather
        # than a raw "found 0 runnable modules" error.
        from torch._precompile import PrecompileError

        x = torch.randn(4)
        for fn in (lambda m, xx: xx, lambda m, xx: xx.detach()):
            with self.assertRaisesRegex(PrecompileError, "no compute"):
                torch.compiler.precompile(fn, torch.nn.Identity(), x)

    def test_cache_none_but_source_runs_when_caches_disabled(self):
        # The cache is purely an acceleration: with caching disabled the artifact has no
        # saveable cache entry (cache is None), yet the emitted source still runs
        # standalone and matches eager. Pins the "cache None but python_code valid"
        # contract that load() relies on for its inlined fallback.
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with inductor_config.patch(force_disable_caches=True):
            src, cache = aot.compile_to_python(gm, _flat_inputs(m, x))
        self.assertIsNone(cache)  # no saveable entry, but the source is still runnable
        call = _exec(src)
        with torch.no_grad():
            out = call(_flat_inputs(m, x))
        self.assertEqual(out[0], m(x))


def _eval_emitted(expr, imports):
    """Evaluate an _emit_value expression the way the generated module would: run the
    recorded imports, define the same _rebuild helper, then eval the expression."""
    ns = {}
    for stmt in imports:
        exec(stmt, ns)

    def _rebuild(obj, state):
        if state is None:
            return obj
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
            return obj
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if state:
            obj.__dict__.update(state)
        if slotstate:
            for k, v in slotstate.items():
                setattr(obj, k, v)
        return obj

    ns["_rebuild"] = _rebuild
    return eval(expr, ns)


class TestEmitValue(TestCase):
    """Unit tests for the pickle-free source emitter behind compile_to_python: every
    baked metadata global is reconstructed as plain source, and a genuinely opaque
    leaf raises rather than falling back to an embedding."""

    def _roundtrip(self, obj):
        from torch._functorch._aot_autograd.to_standalone_python import _emit_value

        imports = set()
        expr = _emit_value(obj, imports)
        self.assertNotIn("_unpickle", expr)
        return _eval_emitted(expr, imports), expr

    def test_primitives_and_containers(self):
        for obj in [None, True, 3, 3.5, "x", b"y", (1, 2), [1, "a"], {1: 2}, {1, 2}]:
            back, _ = self._roundtrip(obj)
            self.assertEqual(back, obj)
        back, _ = self._roundtrip(frozenset({1, 2}))
        self.assertEqual(back, frozenset({1, 2}))

    def test_non_finite_floats(self):
        # repr(float('inf')) is the bare token 'inf', which would NameError in the
        # generated module; the emitter must produce a self-contained constructor.
        import math

        for obj in [float("inf"), float("-inf")]:
            back, expr = self._roundtrip(obj)
            self.assertEqual(back, obj)
            self.assertNotIn("inf,", expr + ",")  # not a bare 'inf' token
        back, _ = self._roundtrip(float("nan"))
        self.assertTrue(math.isnan(back))
        # A complex with a non-finite component round-trips too.
        back, _ = self._roundtrip(complex(float("inf"), 2.0))
        self.assertEqual(back, complex(float("inf"), 2.0))

    def test_torch_scalars(self):
        for obj in [
            torch.float32,
            torch.strided,
            torch.device("cpu"),
            torch.Size([2, 3]),
        ]:
            back, _ = self._roundtrip(obj)
            self.assertEqual(back, obj)

    def test_type_function_enum_partial(self):
        import functools

        from torch._C import _functionalization as _F

        back, _ = self._roundtrip(torch.nn.Linear)
        self.assertIs(back, torch.nn.Linear)
        back, _ = self._roundtrip(_F.InverseReturnMode.ViewOrScatterInverse)
        self.assertEqual(back, _F.InverseReturnMode.ViewOrScatterInverse)
        p = functools.partial(int, base=2)
        back, _ = self._roundtrip(p)
        self.assertEqual(back("101"), 5)

    def test_reduce_based_placements(self):
        # DTensor placement objects are C++ values with no source-friendly constructor;
        # they reconstruct from the pickle reduce state, emitted as readable source.
        from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

        for obj in [Replicate(), Shard(0), Shard(2), Partial(), Partial("avg")]:
            back, expr = self._roundtrip(obj)
            self.assertEqual(back, obj)
            self.assertIn("_rebuild", expr)

    def test_opaque_leaf_raises(self):
        # A lambda and an unpicklable C object are not source-expressible: emit must
        # raise NotImplementedError with a concrete reason, never silently embed. A
        # lambda has no importable qualname (a local definition); threading.Lock has no
        # usable reduce.
        import threading

        from torch._functorch._aot_autograd.to_standalone_python import _emit_value

        with self.assertRaisesRegex(NotImplementedError, "local definition"):
            _emit_value(lambda x: x, set())
        with self.assertRaisesRegex(
            NotImplementedError, "not source-expressible.*reduce"
        ):
            _emit_value(threading.Lock(), set())

    def test_live_tensor_and_storage_raise(self):
        # A live torch.Tensor / torch.UntypedStorage / torch.storage.TypedStorage must
        # never be baked as source: that would embed raw weight bytes and require
        # pickle.loads at exec time. _emit_value rejects all three up front (the blocker
        # fix); the downstream reduce-reject does NOT catch TypedStorage, so its explicit
        # branch needs its own coverage. ``.storage()`` returns a TypedStorage (and warns
        # about deprecation), so suppress that warning while exercising it.
        import warnings

        from torch._functorch._aot_autograd.to_standalone_python import _emit_value

        with self.assertRaisesRegex(NotImplementedError, "live Tensor"):
            _emit_value(torch.randn(3), set())
        with self.assertRaisesRegex(NotImplementedError, "live UntypedStorage"):
            _emit_value(torch.randn(3).untyped_storage(), set())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            typed_storage = torch.randn(3).storage()
        self.assertIsInstance(typed_storage, torch.storage.TypedStorage)
        with self.assertRaisesRegex(NotImplementedError, "live TypedStorage"):
            _emit_value(typed_storage, set())

    def test_set_frozenset_emit_canonical_sorted_order(self):
        # The set/frozenset emitter must be DETERMINISTIC: Python set iteration order is
        # not byte-stable across processes, so the emitted source sorts the elements. The
        # roundtrip test above is order-insensitive; assert the exact canonical form here.
        from torch._functorch._aot_autograd.to_standalone_python import _emit_value

        self.assertEqual(_emit_value({3, 1, 2}, set()), "set([1, 2, 3])")
        self.assertEqual(
            _emit_value(frozenset({3, 1, 2}), set()), "frozenset([1, 2, 3])"
        )
        # A mutually-UNORDERABLE (mixed-type) set cannot sort by value (int < str raises);
        # the emitter falls back to sorting by repr. Assert it does not raise AND is
        # deterministic across two calls (the property a process-stable artifact needs).
        mixed = {1, "a", 2.0}
        first = _emit_value(mixed, set())
        second = _emit_value(mixed, set())
        self.assertEqual(first, second)

    def test_unorderable_set_with_default_repr_element_raises(self):
        # The determinism fix no longer special-cases object.__repr__: an UNORDERABLE
        # (mixed-type) set is sorted by each element's EMITTED SOURCE EXPRESSION, which
        # forces every element through _emit_value's source-expressibility gate. A
        # non-source-expressible element (here a local class with no importable
        # module/qualname) therefore raises via _emit_importable -- naming the missing
        # module/qualname / local definition -- rather than the old memory-address
        # message, since the keying pass itself cannot express it as source.
        from torch._functorch._aot_autograd.to_standalone_python import _emit_value

        class _Opaque:
            pass  # a local definition -> no importable module.qualname

        # Mix with an int so the set is unorderable (int < _Opaque raises), forcing the
        # source-expression-sort path that emits each element and so rejects the local.
        with self.assertRaisesRegex(
            NotImplementedError, "no module/qualname or is a local definition"
        ):
            _emit_value({_Opaque(), 1}, set())

    def test_unorderable_source_expressible_set_emits_content_sorted(self):
        # When an UNORDERABLE (mixed-type) set's elements ARE all source-expressible, the
        # emitter sorts them by their emitted source expression (the exact text that lands
        # in the artifact) rather than failing -- so the output is deterministic and
        # content-sorted across processes. ``int`` < ``str`` is unorderable by value, so
        # this exercises the source-expression-sort fallback on a fully-expressible set.
        from torch._functorch._aot_autograd.to_standalone_python import _emit_value

        mixed = {1, "a", 2.0}
        first = _emit_value(mixed, set())
        second = _emit_value(mixed, set())
        self.assertEqual(first, second)  # deterministic
        # Sorted by emitted source: repr("a") == "'a'" < "1" < "2.0".
        self.assertEqual(first, "set(['a', 1, 2.0])")

    def test_self_referential_container_raises(self):
        # A self-referential (cyclic) metadata container is not expressible as a source
        # literal; the cycle guard raises NotImplementedError rather than recursing until
        # RecursionError.
        from torch._functorch._aot_autograd.to_standalone_python import _emit_value

        cyclic: list = [1, 2]
        cyclic.append(cyclic)
        with self.assertRaisesRegex(NotImplementedError, "self-referential"):
            _emit_value(cyclic, set())


class TestModuleLevelCallDetection(TestCase):
    """Unit tests for the AST-based module-level ``call`` detector and the single-module
    guard behind compile_to_python. These are exercised end-to-end elsewhere, but their
    edge cases (strings/comments/nesting; multi-module / empty-module errors) are pinned
    directly here so a regression in the structural detection surfaces precisely."""

    def test_defines_module_level_call_positive_forms(self):
        # The two codegen'd forms of the runnable entry point are both recognized: a
        # top-level ``def call(`` (graph_partition off) and a top-level
        # ``call = runner.call`` binding (graph_partition on).
        from torch._inductor.standalone_compile import _defines_module_level_call

        self.assertTrue(
            _defines_module_level_call("def call(args):\n    return args\n")
        )
        self.assertTrue(
            _defines_module_level_call("runner = Runner()\ncall = runner.call\n")
        )

    def test_defines_module_level_call_rejects_non_toplevel(self):
        # The detector is STRUCTURAL (AST-parsed), so the false positives a raw substring
        # scan would hit are rejected: a ``def call(`` at column 0 INSIDE a triple-quoted
        # string, a commented-out ``# def call(``, and a nested-only ``def call`` (a class
        # method) with no top-level binding all report no module-level ``call``.
        from torch._inductor.standalone_compile import _defines_module_level_call

        in_string = 'KERNEL = """\ndef call(args):\n    pass\n"""\n'
        self.assertFalse(_defines_module_level_call(in_string))
        self.assertFalse(_defines_module_level_call("# def call(args):\nx = 1\n"))
        nested = "class Runner:\n    def call(self, args):\n        return args\n"
        self.assertFalse(_defines_module_level_call(nested))

    def test_extract_runnable_module_multi_and_empty(self):
        # The single-module guard: two captured modules each defining ``call`` is an
        # unexpected multi-module lowering -> RuntimeError ("exactly one"); an empty
        # capture list (no module defines ``call``) -> NoRunnableInductorModuleError.
        from torch._inductor.standalone_compile import (
            _extract_runnable_module,
            NoRunnableInductorModuleError,
        )

        two = ["def call(args):\n    return 1\n", "def call(args):\n    return 2\n"]
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            _extract_runnable_module(two)
        with self.assertRaises(NoRunnableInductorModuleError):
            _extract_runnable_module([])


if __name__ == "__main__":
    run_tests()
