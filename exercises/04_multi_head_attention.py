"""
练习 4: 多头注意力 (Multi-Head Attention)
==========================================

核心思想: 用多个注意力头并行关注不同的特征子空间，然后拼接起来。
如果单头是"用一种方式看"，多头就是"同时用多种角度看，然后合并"。

nanogpt 参考: train_gpt2.py 中的 CausalSelfAttention 类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参数
BATCH_SIZE = 4
BLOCK_SIZE = 8  # 序列长度
N_EMBD = 64    # 嵌入维度
N_HEAD = 4     # 注意力头数
HEAD_SIZE = N_EMBD // N_HEAD  # 每个头的维度 = 16


class MultiHeadAttention(nn.Module):
    """
    多头因果自注意力

    实现方式: 用一个大的线性层同时计算所有头的 Q, K, V，
    然后 reshape 成多头形式。这比分别建 N_HEAD 个小线性层更高效。
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.n_embd = n_embd

        # TODO 1: 创建 Q, K, V 的投影层
        # 技巧: 用一个线性层 c_attn 同时产生 q, k, v (输出维度 = 3 * n_embd)
        # self.c_attn = nn.Linear(...)

        # TODO 2: 输出投影层，把拼接后的多头输出映射回 n_embd
        # self.c_proj = nn.Linear(...)

        # 因果掩码 (和练习3一样，但这里注册为 buffer)
        self.register_buffer("bias",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)  # (1, 1, T, T) 方便广播
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, time(seq_len), channels(n_embd)

        # TODO 3: 用 self.c_attn 计算 q, k, v
        # qkv = self.c_attn(x)  # (B, T, 3*C)
        # q, k, v = qkv.split(self.n_embd, dim=2)  # 各 (B, T, C)

        # TODO 4: reshape 成多头形式
        # 目标形状: (B, n_head, T, head_size)
        # 提示: 先 view 成 (B, T, n_head, head_size)，再 transpose(1, 2)
        # q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        # k = ...
        # v = ...

        # TODO 5: 计算注意力 (和练习3类似，但现在是4D张量)
        # att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # out = att @ v  # (B, n_head, T, head_size)

        # TODO 6: 把多头拼接回去
        # 提示: transpose 回来，然后 contiguous().view(B, T, C)
        # out = out.transpose(1, 2).contiguous().view(B, T, C)

        # TODO 7: 输出投影
        # out = self.c_proj(out)

        # return out
        pass
