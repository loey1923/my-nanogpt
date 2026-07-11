"""
练习 01: PyTorch 张量基础与 nn.Module
======================================
目标：熟悉 tensor 创建、运算、nn.Module 定义方式
"""
import torch
import torch.nn as nn

print("=== Part 1: 张量创建 ===")

# TODO 1: 创建一个 shape 为 (2, 3) 的全零张量
x = ...
assert x.shape == (2, 3) and x.sum() == 0, "TODO 1 未通过"
print(f"✓ 全零张量: {x.shape}")

# TODO 2: 创建一个从 0 到 9 的一维张量 (提示: torch.arange)
y = ...
assert y.shape == (10,) and y[-1] == 9, "TODO 2 未通过"
print(f"✓ arange 张量: {y}")

# TODO 3: 把 y reshape 成 (2, 5)
z = ...
assert z.shape == (2, 5), "TODO 3 未通过"
print(f"✓ reshape: {z.shape}")

print("\n=== Part 2: 张量运算 ===")

a = torch.randn(3, 4)
b = torch.randn(4, 5)

# TODO 4: 矩阵乘法 a @ b (或 torch.matmul)
c = ...
assert c.shape == (3, 5), "TODO 4 未通过"
print(f"✓ 矩阵乘法: {c.shape}")

# TODO 5: 对 a 沿 dim=1 做 softmax (提示: torch.softmax 或 F.softmax)
import torch.nn.functional as F
a_soft = ...
assert a_soft.shape == (3, 4), "TODO 5 未通过"
assert torch.allclose(a_soft.sum(dim=1), torch.ones(3)), "softmax 每行应和为1"
print(f"✓ softmax: 每行和为1")
