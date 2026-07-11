"""
练习 6: 完整 GPT 模型
把前面所有模块组装成一个完整的语言模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT(nn.Module):
    """
    TODO: 实现完整的 GPT 模型

    结构:
    1. Token Embedding: vocab_size -> n_embd
    2. Position Embedding: block_size -> n_embd
    3. N 个 TransformerBlock
    4. 最终 LayerNorm
    5. Language Model Head: n_embd -> vocab_size (线性层，无bias)

    前向传播:
    - tok_emb = token_embedding(input_ids)
    - pos_emb = position_embedding(positions)
    - x = tok_emb + pos_emb
    - x = blocks(x)  (过 N 个 transformer block)
    - x = final_layernorm(x)
    - logits = lm_head(x)
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size

        # TODO: 定义模型组件
        # self.token_embedding = nn.Embedding(...)
        # self.position_embedding = nn.Embedding(...)
        # self.blocks = nn.ModuleList([...])  # n_layer 个 TransformerBlock
        # self.ln_f = nn.LayerNorm(...)
        # self.lm_head = nn.Linear(..., bias=False)
        pass

    def forward(self, idx, targets=None):
        """
        idx: (B, T) 输入 token ids
        targets: (B, T) 目标 token ids，用于计算 loss

        返回: logits, loss
        - logits: (B, T, vocab_size)
        - loss: 标量，如果 targets 为 None 则返回 None
        """
        B, T = idx.shape

        # TODO: 实现前向传播
        # 1. 生成位置索引: torch.arange(T, device=idx.device)
        # 2. token embedding + position embedding
        # 3. 通过所有 transformer blocks
        # 4. 最终 layernorm
        # 5. lm_head 得到 logits
        # 6. 如果有 targets，计算 cross_entropy loss
        #    注意 reshape: logits (B*T, vocab_size), targets (B*T,)

        logits = None  # 替换为你的实现
        loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        自回归生成

        idx: (B, T) 当前上下文
        max_new_tokens: 要生成的 token 数量

        TODO:
        for _ in range(max_new_tokens):
            1. 裁剪 idx 到最后 block_size 个 token
            2. 前向传播得到 logits
            3. 取最后一个时间步的 logits: logits[:, -1, :]
            4. softmax 得到概率
            5. 采样下一个 token: torch.multinomial
            6. 拼接到 idx 后面
        """
        for _ in range(max_new_tokens):
            pass  # 替换为你的实现
        return idx


# ============ 验证 ============
if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 256
    n_embd = 64
    n_head = 4
    n_layer = 2
    block_size = 32

    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)

    # 测试前向传播
    x = torch.randint(0, vocab_size, (2, 16))
    targets = torch.randint(0, vocab_size, (2, 16))

    logits, loss = model(x, targets)
    print(f"Logits shape: {logits.shape}")  # 应为 (2, 16, 256)
    print(f"Loss: {loss.item():.4f}")  # 随机初始化约为 -ln(1/256) ≈ 5.55

    # 测试生成
    prompt = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")  # 应为 (1, 21)
    print("练习 6 通过!")
