"""验证练习 4"""
import torch
from ex04_multi_head_attention import MultiHeadAttention

BATCH_SIZE = 4
BLOCK_SIZE = 8
N_EMBD = 64
N_HEAD = 4

mha = MultiHeadAttention(N_EMBD, N_HEAD, BLOCK_SIZE)
x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBD)
out = mha(x)

assert out.shape == (BATCH_SIZE, BLOCK_SIZE, N_EMBD), \
    f"输出形状应为 {(BATCH_SIZE, BLOCK_SIZE, N_EMBD)}，得到 {out.shape}"

# 检查因果性: 修改位置5的输入不应影响位置0-4的输出
x2 = x.clone()
x2[:, 5, :] = torch.randn(BATCH_SIZE, N_EMBD)
out2 = mha(x2)
assert torch.allclose(out[:, :5, :], out2[:, :5, :], atol=1e-6), \
    "因果掩码失效: 修改未来位置影响了过去位置的输出"

# 检查参数量
total_params = sum(p.numel() for p in mha.parameters())
expected = (N_EMBD * 3 * N_EMBD + 3 * N_EMBD) + (N_EMBD * N_EMBD + N_EMBD)
assert total_params == expected, \
    f"参数量不对: 期望 {expected}, 得到 {total_params}"

print("✓ 多头注意力练习通过!")
print(f"  参数量: {total_params}")
print(f"  每头维度: {N_EMBD // N_HEAD}")
