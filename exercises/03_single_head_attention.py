"""
练习 03: 从零实现单头 Self-Attention

这是 Transformer 最核心的组件。你需要理解:
- Q, K, V 三个线性变换
- attention score 计算 (缩放点积)
- causal mask (因果掩码，防止看到未来)
- softmax 归一化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 超参数
BATCH_SIZE = 2
BLOCK_SIZE = 8   # 序列长度 T
N_EMBD = 32      # 嵌入维度
HEAD_SIZE = 16   # attention head 的维度


class SingleHeadAttention(nn.Module):
    """单头因果自注意力"""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        # TODO 1: 创建 Q, K, V 三个线性层 (不带 bias)
        # 输入维度 n_embd, 输出维度 head_size
        self.query = ...  # nn.Linear(n_embd, head_size, bias=False)
        self.key = ...
        self.value = ...

        # 注册一个下三角矩阵作为 buffer (不是参数，不参与训练)
        # 这是因果掩码：位置 i 只能看到 <= i 的位置
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        """
        x: shape (B, T, C) - batch, sequence_length, embedding_dim
        返回: shape (B, T, head_size)
        """
        B, T, C = x.shape

        # TODO 2: 计算 Q, K, V
        q = ...  # (B, T, head_size)
        k = ...  # (B, T, head_size)
        v = ...  # (B, T, head_size)

        # TODO 3: 计算 attention scores
        # 公式: (Q @ K^T) / sqrt(head_size)
        # 提示: k.transpose(-2, -1) 转置最后两个维度
        scores = ...  # (B, T, T)

        # TODO 4: 应用因果掩码
        # 把 tril[:T, :T] == 0 的位置设为 -inf
        # 提示: scores.masked_fill(mask == 0, float('-inf'))
        scores = ...

        # TODO 5: softmax 归一化 (在最后一个维度)
        weights = ...  # (B, T, T)

        # TODO 6: 加权求和 value
        out = ...  # (B, T, head_size)
        return out
