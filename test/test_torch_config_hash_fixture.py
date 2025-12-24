# Owner(s): ["module: unknown"]

import difflib

import yaml

from torch._inductor import config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


def get_filtered_config():
    return {
        key: value
        for key, value in inductor_config.save_config_portable().items()
        if not isinstance(value, bool) and value is not None
    }


class TestConfigModule(TestCase):
    def test_inductor_config_hash_portable_fixture(self):
        torch_config = get_filtered_config()
        torch_config_yaml = yaml.dump(
            torch_config,
            sort_keys=True,
        )
        self.assertExpectedInline(
            torch_config_yaml,
            """\
aot_inductor.compile_wrapper_opt_level: O1
aot_inductor.custom_ops_to_c_shims: {}
aot_inductor.debug_intermediate_value_printer: '0'
aot_inductor.metadata: {}
aot_inductor.output_path: ''
aot_inductor.presets: {}
aot_inductor.repro_level: 2
aot_inductor.serialized_in_spec: ''
aot_inductor.serialized_out_spec: ''
aten_distributed_optimizations.collective_estimator: analytical
auto_chunker.amplify_ratio_threshold: 8
auto_chunker.output_size_threshold: 1048576
autoheuristic_collect: ''
autoheuristic_log_path: DEFAULT
autoheuristic_use: mixed_mm
autotune_lookup_table: {}
autotune_num_choices_displayed: 10
bucket_all_gathers_fx: none
bucket_all_reduces_fx: none
bucket_reduce_scatters_fx: none
collective_benchmark_nruns: 50
collective_benchmark_timeout: 30.0
combo_kernel_allow_mixed_sizes: 1
combo_kernel_max_num_args: 250
combo_kernels_autotune: 1
coordinate_descent_search_radius: 1
cpp.cxx: !!python/tuple
- null
- g++
cpp.descriptive_names: original_aten
cpp.enable_floating_point_contract_flag: 'off'
cpp.gemm_max_k_slices: 1
cpp.max_horizontal_fusion_size: 16
cpp.min_chunk_size: 512
cpp.threads: -1
cpu_backend: cpp
cpu_gpu_bw: 50.0
cuda.compile_opt_level: -O1
cuda.cutlass_backend_min_gemm_size: 1
cuda.cutlass_enabled_ops: all
cuda.cutlass_instantiation_level: '0'
cuda.cutlass_max_profiling_swizzle_options:
- 1
- 2
- 4
- 8
cuda.nvgemm_max_profiling_configs: 5
cuda_backend: triton
custom_should_partition_ops: []
enabled_metric_tables: ''
estimate_op_runtime: default
external_matmul: []
file_lock_timeout: 600
fx_passes_numeric_check:
  num_iterations: 1
  pre_grad: false
  precision: 0.0001
  requires_optimizer: true
halide.cpu_target: host
halide.gpu_target: host-cuda
halide.scheduler_cpu: Adams2019
halide.scheduler_cuda: Anderson2021
inductor_default_autotune_rep: 100
inductor_default_autotune_warmup: 25
inter_node_bw: 25
intra_node_bw: 300
kernel_name_max_ops: 10
layout_opt_default: '1'
max_autotune_conv_backends: ATEN,TRITON
max_autotune_flex_search_space: DEFAULT
max_autotune_gemm_backends: ATEN,TRITON,CPP
max_autotune_gemm_search_space: DEFAULT
max_autotune_subproc_graceful_timeout_seconds: 0.0
max_autotune_subproc_result_timeout_seconds: 60.0
max_autotune_subproc_terminate_timeout_seconds: 0.0
max_epilogue_benchmarked_choices: 1
max_fusion_buffer_group_pairwise_attempts: 64
max_fusion_size: 64
max_pointwise_cat_inputs: 8
memory_pool: intermediates
min_num_split: 0
mixed_mm_choice: heuristic
multi_kernel_hints: []
padding_alignment_bytes: 128
padding_stride_threshold: 1024
post_grad_fusion_options: {}
pre_grad_fusion_options: {}
precompilation_timeout_seconds: 3600
profile_bandwidth_regex: ''
quiesce_async_compile_time: 60
realize_acc_reads_threshold: 8
realize_opcount_threshold: 30
realize_reads_threshold: 4
reorder_for_compute_comm_overlap_passes: []
rocm.arch: []
rocm.ck_supported_arch:
- gfx90a
- gfx942
- gfx950
rocm.compile_opt_level: -O2
rocm.contiguous_threshold: 16
rocm.split_k_threshold: 16
score_fusion_memory_threshold: 10
size_threshold_for_succ_based_strategy: 0
small_memory_access_threshold: 16777216
test_configs.distort_benchmarking_result: ''
torchinductor_worker_logpath: ''
triton.cudagraph_dynamic_shape_warn_limit: 8
triton.cudagraph_unexpected_rerecord_limit: 128
triton.debug_dump_kernel_inputs: {}
triton.decompose_k_threshold: 32
triton.descriptive_names: original_aten
triton.max_kernel_dump_occurrences: 3
triton.min_split_scan_rblock: 256
triton.mix_order_reduction_initial_xblock: 1
triton.multi_kernel: 0
triton.num_decompose_k_splits: 10
triton.spill_threshold: 16
triton_kernel_default_layout_constraint: needs_fixed_stride_order
unbacked_symint_fallback: 8192
unroll_reductions_threshold: 8
unsafe_marked_cacheable_functions: {}
xpu_backend: triton
""",
        )

    def test_inductor_config_hash_portable_without_ignore(self):
        """
        Detect the inductor config hash will change if we forgot to ignore cuda.cutlass_dir.
        """
        expected_torch_config = get_filtered_config()
        expected_torch_config_yaml = yaml.dump(
            expected_torch_config,
            sort_keys=True,
        )

        idx = inductor_config._cache_config_ignore_prefix.index("cuda.cutlass_dir")
        inductor_config._cache_config_ignore_prefix.remove("cuda.cutlass_dir")
        try:
            changed_torch_config = get_filtered_config()
            changed_torch_config_yaml = yaml.dump(
                changed_torch_config,
                sort_keys=True,
            )
            self.assertNotEqual(changed_torch_config_yaml, expected_torch_config_yaml)
            diff = difflib.ndiff(
                expected_torch_config_yaml.splitlines(keepends=True),
                changed_torch_config_yaml.splitlines(keepends=True),
            )
            diff_lines = [line for line in diff if line.startswith(("+ ", "- "))]
            self.assertEqual(len(diff_lines), 1)
            self.assertTrue(diff_lines[0].startswith("+ cuda.cutlass_dir: "))
        finally:
            inductor_config._cache_config_ignore_prefix.insert(idx, "cuda.cutlass_dir")


if __name__ == "__main__":
    run_tests()
