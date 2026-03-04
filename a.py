import torch
import torch._inductor.config as inductor_config

@torch.compile
def f(x, y):
    a = x.sum(dim=-1, keepdim=True)
    b = x.sum(dim=0, keepdim=True)
    x = a + b
    z = x @ y
    z = z + 1
    z = z @ y
    return z, a, b

x = torch.randn(1024 * 32, 1024, device="cuda", dtype=torch.bfloat16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
f(x, y)
