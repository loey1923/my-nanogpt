"""
练习 5: Transformer Block
=========================

一个完整的 Transformer block 包含:
1. LayerNorm + Multi-Head Attention + 残差连接
2. LayerNorm + FFN (Feed-Forward Network) + 残差连接

FFN 结构: Linear(n_embd, 4*n_embd) -> GELU -> Linear(4*n_embd, n_embd)

注意 nanogpt 用的是 Pre-Norm 结构 (先 norm 再 attention/ffn),
不是原始论文的 Post-Norm。

关键点:
- 残差连接: output = x + sublayer(norm(x))
- FFN 的隐藏层是 4 倍扩展
- GELU 激活函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """从练习4复用，这里给你完整版本"""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class FeedForward(nn.Module):
    """
    TODO: 实现 FFN 子层
    结构: Linear(n_embd, 4*n_embd) -> GELU -> Linear(4*n_embd, n_embd)
    """
    def __init__(self, n_embd):
        super().__init__()
        # TODO: 定义两个线性层
        # self.c_fc = ...     (n_embd -> 4*n_embd)
        # self.c_proj = ...   (4*n_embd -> n_embd)
        pass

    def forward(self, x):
        # TODO: fc -> gelu -> proj
        # 提示: F.gelu(x) 或 nn.GELU()
        pass


class TransformerBlock(nn.Module):
    """
    TODO: 实现完整的 Transformer block
    结构 (Pre-Norm):
        x = x + attention(layernorm1(x))
        x = x + ffn(layernorm2(x))
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        # TODO: 创建以下组件:
        # self.ln_1 = LayerNorm
        # self.attn = MultiHeadAttention
        # self.ln_2 = LayerNorm
        # self.ffn = FeedForward
        pass

    def forward(self, x):
        # TODO: 实现 pre-norm 结构的前向传播
        # 记住残差连接!
        pass


# ============ 验证 ============
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, C = 2, 8, 64
    n_head = 4
    block_size = 32

    block = TransformerBlock(n_embd=C, n_head=n_head, block_size=block_size)
    x = torch.randn(B, T, C)
    out = block(x)

    assert out.shape == (B, T, C), f"输出形状错误: {out.shape}"

    # 验证残差连接存在: 如果去掉残差，输出不会接近输入
    # 简单检查: 输出不应该和输入完全无关
    print(f"输入均值: {x.mean():.4f}, 输出均值: {out.mean():.4f}")
    print(f"输入输出差的范数: {(out - x).norm():.4f}")
    print("Transformer Block 验证通过!")
