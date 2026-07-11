"""
练习 01 (续): nn.Module 与自动求导
===================================
目标：理解 nn.Module 的参数管理和 autograd
"""
import torch
import torch.nn as nn

print("=== Part 3: 自动求导 ===")

# TODO 6: 创建一个 requires_grad=True 的张量 w，值为 [2.0, 3.0]
w = ...
assert w.requires_grad, "TODO 6: 需要 requires_grad=True"

# TODO 7: 计算 loss = (w * 3).sum()，然后调用 loss.backward()
loss = ...
# 调用 backward
...
assert w.grad is not None, "TODO 7: backward 后 grad 不应为 None"
assert torch.allclose(w.grad, torch.tensor([3.0, 3.0])), "梯度应为 [3, 3]"
print(f"✓ 自动求导: w.grad = {w.grad}")

print("\n=== Part 4: nn.Module ===")


# TODO 8: 实现一个简单的线性层模块
# 输入 (batch, in_features) -> 输出 (batch, out_features)
# 包含 self.linear = nn.Linear(in_features, out_features)
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO: 定义 self.linear
        ...

    def forward(self, x):
        # TODO: 返回 self.linear(x)
        ...


model = MyLinear(4, 2)
params = list(model.parameters())
assert len(params) == 2, "应有 weight 和 bias 两个参数"
test_input = torch.randn(3, 4)
output = model(test_input)
assert output.shape == (3, 2), "输出 shape 不对"
print(f"✓ nn.Module: 参数数量={len(params)}, 输出shape={output.shape}")

print("\n=== Part 5: Optimizer ===")

# TODO 9: 创建 AdamW 优化器，学习率 0.01
optimizer = ...  # 提示: torch.optim.AdamW(model.parameters(), lr=...)

# 模拟一步训练
target = torch.randn(3, 2)
loss = ((model(test_input) - target) ** 2).mean()
loss.backward()

# TODO 10: 执行一步优化 (optimizer.step()) 并清零梯度 (optimizer.zero_grad())
...

print(f"✓ 优化器: loss={loss.item():.4f}")
print("\n🎉 01_tensor_basics 全部完成!")
