# mypy: allow-untyped-defs
from collections.abc import Sequence
from typing import Any, Final, Literal, TypeAlias

from torch.utils._ordered_set import OrderedSet


LOCAL_REDUCE_COMPRESSED_AUX: Final = "compressed_aux"
LOCAL_REDUCE_FEED_MAIN: Final = "feed_main"
LOCAL_REDUCE_FEED_MAIN_ARG_NAME: Final = "local_reduce0"
LOCAL_REDUCE_COMBINE_FN_SUFFIX: Final = "_local_reduce_combine_fn"
LOCAL_REDUCE_FINALIZE_FN_SUFFIX: Final = "_local_reduce_finalize_fn"
LOCAL_REDUCE_COMBINE_KEY_SUFFIX: Final = ":local_reduce_combine"
LOCAL_REDUCE_FINALIZE_KEY_SUFFIX: Final = ":local_reduce_finalize"
LOCAL_REDUCE_RETURNS_KWARG: Final = "tensor_epilogue_returns_local_reduce"
LOCAL_REDUCE_OUT_KWARG: Final = "local_reduce_out"
LOCAL_REDUCE_GROUP_KWARG: Final = "local_reduce_group"
LOCAL_REDUCE_AXIS_KWARG: Final = "local_reduce_axis"
LOCAL_REDUCE_FEEDS_MAIN_KWARG: Final = "local_reduce_feeds_main"
LOCAL_REDUCE_COMBINE_FN_KWARG: Final = "local_reduce_combine_fn"
LOCAL_REDUCE_COMBINE_KEY_KWARG: Final = "local_reduce_combine_key"
LOCAL_REDUCE_FINALIZE_FN_KWARG: Final = "local_reduce_finalize_fn"
LOCAL_REDUCE_FINALIZE_KEY_KWARG: Final = "local_reduce_finalize_key"
FlexGemmLocalReduceConsumerKind: TypeAlias = Literal["compressed_aux", "feed_main"]


def normalize_reduce_dims(dim: Any) -> tuple[Any, ...]:
    """Normalize scalar and tuple/list reduction dims for grouped-domain checks."""
    return tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)


def grouped_reduce_dims_match(dim: Any, reduce_dims: Sequence[Any]) -> bool:
    """Treat equivalent grouped-dimension spellings as one reduction domain."""
    return all(item in reduce_dims for item in normalize_reduce_dims(dim))


# Feed-main currently reduces only within one lane-layout M group; cross-warp M
# stitching needs the two-phase/replay path used by compressed aux reductions.
MAX_SAME_WARP_LOCAL_REDUCE_FEED_MAIN_GROUP = 16
LOCAL_REDUCE_FEED_MAIN_AXIS_ERROR = (
    "FlexGEMM local-reduce feed-main currently supports only axis 0"
)
LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR = (
    "FlexGEMM local-reduce feed-main currently supports only same-warp axis-0 "
    f"groups <= {MAX_SAME_WARP_LOCAL_REDUCE_FEED_MAIN_GROUP}"
)
LOCAL_REDUCE_DIVISIBLE_SHAPE_ERROR = (
    "local_reduce_group must divide the selected FlexGEMM output dimension"
)
LOCAL_REDUCE_GROUP_POSITIVE_ERROR = "local_reduce_group must be positive"
LOCAL_REDUCE_AXIS_ERROR = "local_reduce_axis must be 0 or 1"
LOCAL_REDUCE_CONSUMER_KIND_ERROR = "invalid local-reduce consumer kind"
LOCAL_REDUCE_TENSORSSA_GROUP_SIZE_ERROR = (
    "FlexGEMM local reductions require group size greater than 1"
)
LOCAL_REDUCE_TENSORSSA_FRAGMENT_MULTIPLE_ERROR = (
    "FlexGEMM local reductions larger than TensorSSA fragment width 32 "
    "require group size to be a multiple of 32"
)
LOCAL_REDUCE_TENSORSSA_FRAGMENT_DIVISIBLE_ERROR = (
    "FlexGEMM local reductions require group size to divide TensorSSA fragment width 32"
)
LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR = (
    "unsupported FlexGEMM epilogue partial-output contract: reductions "
    "over GEMM M/N dimensions require an explicit partial-output contract; "
    "QuACK row/column partial reductions produce CTA-tiled partial shapes, "
    "not final PyTorch reduction tensors"
)
LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR = (
    "FlexGEMM local reductions do not support mixing grouped TensorSSA "
    "values with different grouped layouts"
)
LOCAL_REDUCE_DENSE_MM_SCOPE_ERROR = (
    "FlexGEMM local reductions currently support only aten.mm"
)
LOCAL_REDUCE_C_ALPHA_BETA_ERROR = (
    "FlexGEMM local reductions cannot be combined with C/alpha/beta yet"
)
LOCAL_REDUCE_SWAP_AB_ERROR = (
    "FlexGEMM local reductions do not support swap_ab configs yet"
)
LOCAL_REDUCE_AUX_OUT_COMPOSITION_ERROR = (
    "FlexGEMM local-reduce aux outputs cannot be combined with aux_out yet"
)
LOCAL_REDUCE_AUX_SAME_SHAPE_COMPOSITION_ERROR = "FlexGEMM local-reduce aux outputs cannot be combined with same-shape aux outputs yet"
LOCAL_REDUCE_AUX_METADATA_ERROR = (
    "FlexGEMM local-reduce aux outputs require aux output metadata"
)
LOCAL_REDUCE_AUX_TENSORSSA_ERROR = (
    "FlexGEMM local-reduce aux output must be produced by a grouped TensorSSA reduction"
)
LOCAL_REDUCE_AUX_OUTPUT_CONTRACT_ERROR = (
    "FlexGEMM generic aux tuple epilogues currently require aux output shapes "
    "to match the GEMM output shape; compressed or block reductions across GEMM "
    "M/N dimensions require an explicit local-reduce output contract"
)
LOCAL_REDUCE_GROUPED_TENSORSSA_LOWERING_ERROR = (
    "unsupported FlexGEMM epilogue local reduction without a grouped TensorSSA lowering"
)
LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR = (
    "FlexGEMM local-reduce broadcast values support one generated physical reduction"
)
LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR = (
    "FlexGEMM physical finalize expressions support a single physical local reduction"
)
LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR = (
    "unsupported FlexGEMM physical local reduction: post-reduction pointwise transforms "
    "require generated finalize code"
)
LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR = (
    "unsupported FlexGEMM physical local reduction: finalize expressions must "
    "depend only on the reduced value and scalar constants"
)
LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR = (
    "FlexGEMM physical local-reduce feed-main source expressions require "
    "two-phase local-reduce source lowering"
)
LOCAL_REDUCE_CONFIG_ERROR = (
    "FlexGEMM local-reduce aux outputs require a non-swap_ab "
    "32-lane epilogue-fragment config whose CTA tile axis is divisible by group"
)
LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR = (
    "unsupported FlexGEMM epilogue local reduction: explicit reduction dtype"
)
LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR = (
    "unsupported FlexGEMM epilogue local reduction: currently support only "
    "the innermost grouped dimension"
)
LOCAL_REDUCE_MOMENT_PHYSICAL_COMBINE_ERROR = (
    "unsupported FlexGEMM physical local reduction: moment reductions "
    "need matching QuACK combine/finalize callbacks"
)
LOCAL_REDUCE_MOMENT_INNERMOST_GROUPED_DIM_ERROR = (
    "unsupported FlexGEMM epilogue local reduction: moment reductions "
    "currently support only the innermost grouped dimension"
)
LOCAL_REDUCE_MOMENT_DYNAMIC_CORRECTION_ERROR = (
    "unsupported FlexGEMM epilogue local reduction: dynamic variance correction"
)
LOCAL_REDUCE_MOMENT_CORRECTION_RANGE_ERROR = (
    "unsupported FlexGEMM epilogue local reduction: variance correction "
    "must be smaller than the group size"
)
LOCAL_REDUCE_PREPARE_SOFTMAX_PHYSICAL_COMBINE_ERROR = (
    "unsupported FlexGEMM physical local reduction: prepare_softmax_online "
    "needs a multi-value generated physical reducer"
)
LOCAL_REDUCE_PREPARE_SOFTMAX_GROUPED_DIM_ERROR = (
    "unsupported FlexGEMM epilogue local reduction: prepare_softmax_online "
    "currently supports only the grouped dimension"
)
LOCAL_REDUCE_MIXED_CONTRACT_ERROR = (
    "FlexGEMM local reductions do not support mixing different local-reduce contracts"
)
LOCAL_REDUCE_FEED_MAIN_MIXED_CONTRACT_ERROR = (
    "FlexGEMM local-reduce broadcast values do not support mixing different "
    "local-reduce contracts"
)
FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR = "FlexGEMM output plans require tensor output nodes"
FLEX_GEMM_OUTPUT_TENSOR_ERROR = "FlexGEMM expects tensor outputs"
LOCAL_REDUCE_CONTRACT_NODE_ERROR = "local-reduce contracts require tensor nodes"
LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR = "local-reduce output plans require tensor nodes"
LOCAL_REDUCE_MATERIALIZATION_CONSUMER_ERROR = (
    "local-reduce materialization supports one consumer"
)
LOCAL_REDUCE_MATERIALIZATION_FEED_MAIN_INPUT_ERROR = (
    "feed_main_input requires feed_main"
)
LOCAL_REDUCE_RUNTIME_OUT_ERROR = "compressed local reductions require local_reduce_out"
LOCAL_REDUCE_RUNTIME_FEED_MAIN_OUT_ERROR = (
    "feed-main local reductions cannot store local_reduce_out"
)
LOCAL_REDUCE_RUNTIME_DENSE_MM_ERROR = (
    "FlexGEMM local-reduce {kind} currently supports only 2-D aten.mm"
)
LOCAL_REDUCE_OUT_SHAPE_ERROR = "local_reduce_out shape must be {expected}, got {actual}"
LOCAL_REDUCE_TEMPLATE_OUT_INDEX_ERROR = (
    "compressed local-reduce stores require out_index"
)
LOCAL_REDUCE_TEMPLATE_FEED_MAIN_OUT_INDEX_ERROR = (
    "feed-main local reductions cannot have out_index"
)
LOCAL_REDUCE_GROUP_AXIS_REQUIRED_ERROR = (
    "local_reduce_group and local_reduce_axis must be set"
)
LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR = (
    "physical local reductions require generated local-reduce callbacks"
)
LOCAL_REDUCE_CONSUMER_KINDS = (LOCAL_REDUCE_COMPRESSED_AUX, LOCAL_REDUCE_FEED_MAIN)


def local_reduce_combine_fn_name(epilogue_name: str) -> str:
    """Return the generated CuTeDSL callback name for physical reduction combine."""
    return f"{epilogue_name}{LOCAL_REDUCE_COMBINE_FN_SUFFIX}"


def local_reduce_finalize_fn_name(epilogue_name: str) -> str:
    """Return the generated CuTeDSL callback name for physical reduction finalize."""
    return f"{epilogue_name}{LOCAL_REDUCE_FINALIZE_FN_SUFFIX}"


def local_reduce_default_combine_key(epilogue_key: str) -> str:
    """Return the runtime cache key for an unnamed generated combine callback."""
    return f"{epilogue_key}{LOCAL_REDUCE_COMBINE_KEY_SUFFIX}"


def local_reduce_default_finalize_key(epilogue_key: str) -> str:
    """Return the runtime cache key for an unnamed generated finalize callback."""
    return f"{epilogue_key}{LOCAL_REDUCE_FINALIZE_KEY_SUFFIX}"


def local_reduce_partial_output_contract_error() -> NotImplementedError:
    """Build the shared error for reductions that need partial-output contracts."""
    return NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)


def is_flex_gemm_partial_reduction_shape(
    aux_size: Sequence[Any], output_size: Sequence[Any]
) -> bool:
    """Recognize final M/N reductions that are not QuACK partial outputs."""
    if len(output_size) != 2:
        return False
    aux_shape = list(aux_size)
    m, n = output_size
    if aux_shape in ([], [m], [n], [1, 1], [m, 1], [1, n]):
        return True
    if not all(isinstance(dim, int) for dim in (*aux_shape, m, n)):
        return False
    if len(aux_shape) != 2:
        return False
    aux_m, aux_n = aux_shape
    return (
        aux_m > 0
        and aux_n > 0
        and aux_m <= m
        and aux_n <= n
        and (aux_m < m or aux_n < n)
        and m % aux_m == 0
        and n % aux_n == 0
    )


def local_reduce_grouped_tensorssa_lowering_error(target: Any) -> NotImplementedError:
    """Build the shared fallback error for reductions that reached scalar lowering."""
    return NotImplementedError(
        f"{LOCAL_REDUCE_GROUPED_TENSORSSA_LOWERING_ERROR}: {target}"
    )


def local_reduce_unsupported_tensorssa_error(
    reduction: Any, *, value_only: bool = False
) -> NotImplementedError:
    """Build the shared error for reductions that lack a TensorSSA lowering."""
    suffix = " value-only reduction" if value_only else ""
    return NotImplementedError(
        "unsupported FlexGEMM epilogue local reduction: "
        f"{reduction} does not map to a CuTe TensorSSA{suffix}"
    )


def local_reduce_unsupported_tensorssa_reduction_error(
    reduction_type: Any,
) -> NotImplementedError:
    """Build the shared fallback for reductions without a TensorSSA lowering."""
    return NotImplementedError(
        "unsupported FlexGEMM epilogue local reduction: "
        f"reduction_type={reduction_type!r}"
    )


def local_reduce_unsupported_physical_reduction_error(
    reduction_type: Any,
) -> NotImplementedError:
    """Build the shared fallback for reductions without physical combine support."""
    return NotImplementedError(
        "unsupported FlexGEMM local reduction: "
        f"reduction_type={reduction_type!r} needs a generated physical reducer"
    )


def validate_local_reduce_consumer_kind(kind: str) -> None:
    """Reject unknown local-reduce consumer tags."""
    if kind not in LOCAL_REDUCE_CONSUMER_KINDS:
        raise RuntimeError(f"{LOCAL_REDUCE_CONSUMER_KIND_ERROR}: {kind}")


def local_reduce_feeds_main(kind: FlexGemmLocalReduceConsumerKind) -> bool:
    """Return whether a local-reduce consumer feeds the generated main epilogue."""
    return kind == LOCAL_REDUCE_FEED_MAIN


def local_reduce_stores_compressed_aux(kind: FlexGemmLocalReduceConsumerKind) -> bool:
    """Return whether a local-reduce consumer stores a compressed aux output."""
    return kind == LOCAL_REDUCE_COMPRESSED_AUX


def local_reduce_consumer_kind(*, feeds_main: bool) -> FlexGemmLocalReduceConsumerKind:
    """Map runtime feed-main intent to the tagged local-reduce consumer kind."""
    return LOCAL_REDUCE_FEED_MAIN if feeds_main else LOCAL_REDUCE_COMPRESSED_AUX


def validate_local_reduce_group_axis(group: int, axis: int) -> None:
    """Reject invalid local-reduce geometry common to all plan layers."""
    if group <= 0:
        raise RuntimeError(LOCAL_REDUCE_GROUP_POSITIVE_ERROR)
    if axis not in (0, 1):
        raise RuntimeError(LOCAL_REDUCE_AXIS_ERROR)


def require_local_reduce_group_axis(
    group: int | None, axis: int | None
) -> tuple[int, int]:
    """Normalize optional runtime local-reduce geometry into validated integers."""
    if group is None or axis is None:
        raise RuntimeError(LOCAL_REDUCE_GROUP_AXIS_REQUIRED_ERROR)
    validate_local_reduce_group_axis(group, axis)
    return group, axis


def validate_local_reduce_callbacks(combine_fn: Any, finalize_fn: Any) -> None:
    """Require generated callbacks whenever QuACK needs physical reduction code."""
    if combine_fn is None or finalize_fn is None:
        raise RuntimeError(LOCAL_REDUCE_CALLBACKS_REQUIRED_ERROR)


def validate_local_reduce_output_binding(
    kind: FlexGemmLocalReduceConsumerKind,
    has_output_binding: bool,
    *,
    compressed_missing_error: str,
    feed_main_unexpected_error: str,
) -> None:
    """Reject consumer/output binding combinations shared across plan layers."""
    if local_reduce_stores_compressed_aux(kind) and not has_output_binding:
        raise RuntimeError(compressed_missing_error)
    if local_reduce_feeds_main(kind) and has_output_binding:
        raise RuntimeError(feed_main_unexpected_error)


def validate_local_reduce_selected_dim_divisible(
    shape: Sequence[Any], group: int, axis: int
) -> None:
    """Reject grouped reductions that cannot evenly compress the selected dimension."""
    validate_local_reduce_group_axis(group, axis)
    if shape[axis - 2] % group != 0:
        raise RuntimeError(LOCAL_REDUCE_DIVISIBLE_SHAPE_ERROR)


def validate_local_reduce_tensorssa_group_size(axis: int, group: int) -> None:
    """Reject grouped TensorSSA layouts that cannot map to 32-lane fragments."""
    if group <= 1:
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_GROUP_SIZE_ERROR)
    validate_local_reduce_group_axis(group, axis)
    if group > 32 and group % 32 != 0:
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_FRAGMENT_MULTIPLE_ERROR)
    if group <= 32 and 32 % group != 0:
        raise NotImplementedError(LOCAL_REDUCE_TENSORSSA_FRAGMENT_DIVISIBLE_ERROR)


def validate_local_reduce_runtime_dense_mm(
    kind: FlexGemmLocalReduceConsumerKind, ndim: int
) -> None:
    """Keep runtime local-reduce dispatch scoped to dense 2-D GEMMs."""
    if ndim != 2:
        raise NotImplementedError(LOCAL_REDUCE_RUNTIME_DENSE_MM_ERROR.format(kind=kind))


def validate_local_reduce_out_shape(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> None:
    """Reject compressed local-reduce outputs with the wrong runtime shape."""
    actual = tuple(actual_shape)
    expected = tuple(expected_shape)
    if actual != expected:
        raise RuntimeError(
            LOCAL_REDUCE_OUT_SHAPE_ERROR.format(expected=expected, actual=actual)
        )


def validate_local_reduce_feed_main_same_warp_group(group: int) -> None:
    """Gate current feed-main value availability to same-warp grouped-M cases."""
    if group > MAX_SAME_WARP_LOCAL_REDUCE_FEED_MAIN_GROUP:
        raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_SAME_WARP_ERROR)


def local_reduce_feed_main_supported(axis: int, group: int) -> bool:
    """Return whether current single-pass physical feed-main can produce a value."""
    return axis == 0 and group <= MAX_SAME_WARP_LOCAL_REDUCE_FEED_MAIN_GROUP


def validate_local_reduce_feed_main_capability(axis: int, group: int) -> None:
    """Raise the public error for unsupported current physical feed-main cases."""
    if axis != 0:
        raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS_ERROR)
    validate_local_reduce_feed_main_same_warp_group(group)


def local_reduce_compressed_shape(
    shape: Sequence[Any], group: int, axis: int
) -> tuple[Any, ...]:
    """Return the output shape after compressing the selected GEMM dimension."""
    validate_local_reduce_group_axis(group, axis)
    result = list(shape)
    result[axis - 2] //= group
    return tuple(result)


def validate_local_reduce_no_c_alpha_beta(
    effective_C: Any | None, alpha: float, beta: float
) -> None:
    """Reject C/alpha/beta composition until local-reduce ordering is explicit."""
    if effective_C is not None or alpha != 1.0 or beta != 1.0:
        raise NotImplementedError(LOCAL_REDUCE_C_ALPHA_BETA_ERROR)


def validate_local_reduce_no_aux_out_composition(aux_out: Any | None) -> None:
    """Reject mixing compressed local-reduce aux stores with same-shape aux stores."""
    if aux_out is not None:
        raise NotImplementedError(LOCAL_REDUCE_AUX_OUT_COMPOSITION_ERROR)


def flex_gemm_local_reduce_config_fields(
    config: Any,
) -> tuple[bool, int, int, int, int]:
    """Normalize config objects and keys for local-reduce capability checks."""
    if isinstance(config, dict):
        return (
            config["swap_ab"],
            config["tile_m"],
            config["tile_n"],
            config["cluster_m"],
            config["cluster_n"],
        )
    return (
        config.swap_ab,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
    )


def validate_flex_gemm_local_reduce_config(config: Any, group: int, axis: int) -> bool:
    """Return whether a QuACK config can keep grouped reductions inside one CTA."""
    swap_ab, tile_m, tile_n, cluster_m, cluster_n = (
        flex_gemm_local_reduce_config_fields(config)
    )
    if axis not in (0, 1) or swap_ab:
        return False
    if tile_n < 128 or tile_n % 64 != 0:
        return False
    tile = tile_n if axis == 1 else tile_m
    if tile % group != 0:
        return False
    if group <= 32:
        return 32 % group == 0 and group < tile
    return (
        group % 32 == 0
        and group <= tile
        and tile_m == 128
        and cluster_m == 1
        and cluster_n == 1
    )


def max_flex_gemm_local_reduce_group_for_configs(
    configs: Sequence[Any], axis: int
) -> int | None:
    """Return the largest group accepted by the current local-reduce config gate."""
    candidates: OrderedSet[int] = OrderedSet()
    for config in configs:
        swap_ab, tile_m, tile_n, cluster_m, cluster_n = (
            flex_gemm_local_reduce_config_fields(config)
        )
        if axis not in (0, 1) or swap_ab or tile_n < 128 or tile_n % 64 != 0:
            continue
        tile = tile_n if axis == 1 else tile_m
        for group in (2, 4, 8, 16, 32):
            if group < tile and tile % group == 0:
                candidates.add(group)
        if tile_m == 128 and cluster_m == 1 and cluster_n == 1:
            for group in range(64, tile + 1, 32):
                if tile % group == 0:
                    candidates.add(group)
    return max(candidates) if candidates else None


def flex_gemm_local_reduce_config_error(
    configs: Sequence[Any], group: int, axis: int
) -> str:
    """Explain the current config-filter frontier for local-reduce groups."""
    max_group = max_flex_gemm_local_reduce_group_for_configs(configs, axis)
    if max_group is None:
        return LOCAL_REDUCE_CONFIG_ERROR
    return (
        f"{LOCAL_REDUCE_CONFIG_ERROR}; requested group={group}, "
        f"max supported group={max_group} for axis={axis}"
    )


def local_reduce_needs_physical_callbacks(
    kind: FlexGemmLocalReduceConsumerKind, axis: int, group: int
) -> bool:
    """Return whether QuACK needs generated combine/finalize callbacks."""
    return local_reduce_feeds_main(kind) or axis == 0 or group > 32
