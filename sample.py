import torch
from torch._C import DispatchKey, _IncludeDispatchKeyGuard
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorConverter
from torch.fx.experimental.symbolic_shapes import ShapeEnv

shape_env = ShapeEnv()
converter = FakeTensorConverter()

# --- Concrete shapes (no source) ---
real_a = torch.randn(3, 4)
real_b = torch.randn(3, 4)
fake_a = torch._C._make_fake_tensor(real_a, converter, shape_env)
fake_b = torch._C._make_fake_tensor(real_b, converter, shape_env)

print(f"fake_a.device = {fake_a.device}")               # cpu
print(f"fake_a.shape = {fake_a.shape}")                  # torch.Size([3, 4])
print(f"is_fake = {torch._C._is_fake_tensor(fake_a)}")  # True

with _IncludeDispatchKeyGuard(DispatchKey.Fake):
    c = fake_a + fake_b

print(f"c.device = {c.device}")                          # cpu
print(f"c.shape = {c.shape}")                            # torch.Size([3, 4])
print(f"is_fake = {torch._C._is_fake_tensor(c)}")       # True

# --- Dynamic shapes (with source) ---
real_x = torch.randn(5, 8)
real_y = torch.randn(5, 8)
fake_x = torch._C._make_fake_tensor(
    real_x, converter, shape_env, source=LocalSource("x"))
fake_y = torch._C._make_fake_tensor(
    real_y, converter, shape_env, source=LocalSource("y"))

print(f"\nfake_x.shape = {fake_x.shape}")               # symbolic, e.g. [s0, s1]
print(f"fake_y.shape = {fake_y.shape}")                  # same symbols (shared)
print(f"fake_x.shape[0].node = {fake_x.shape[0].node}") # SymInt node

# Shape arithmetic works through ShapeEnv
s0 = fake_x.shape[0]
s1 = fake_x.shape[1]
print(f"s0 + s1 = {s0 + s1}")                           # symbolic add
print(f"s0 * 2 = {s0 * 2}")                             # symbolic mul
print(f"s0 == fake_y.shape[0]: {s0 == fake_y.shape[0]}") # True (same symbol)
