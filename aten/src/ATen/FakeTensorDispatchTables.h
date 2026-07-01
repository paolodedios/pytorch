#pragma once

#include <ATen/core/operator_name.h>
#include <c10/macros/Export.h>

namespace at::impl {

enum class FakeDispatchCategory { Decomp, Meta, OpImpl, PrimMeta };

TORCH_API void fakeDispatchTableAdd(
    FakeDispatchCategory category,
    const c10::OperatorName& name);
TORCH_API void fakeDispatchTableRemove(
    FakeDispatchCategory category,
    const c10::OperatorName& name);

TORCH_API bool fakeHasUsableDecomp(const c10::OperatorName& name);
TORCH_API bool fakeHasPrimMeta(const c10::OperatorName& name);
TORCH_API bool fakeHasOpImpl(const c10::OperatorName& name);

} // namespace at::impl
