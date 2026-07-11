"""
练习 8: 文本生成 (Autoregressive Sampling)

模型训练好之后，用它来逐 token 生成文本。
这是 GPT 推理的核心逻辑。
"""
import torch
import torch.nn.functional as F


# 假设你的模型已经训练好，这里练习生成逻辑

def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    """
    自回归生成。

    参数:
        model: 训练好的 GPT 模型
        idx: (B, T) 当前上下文 token ids
        max_new_tokens: 要生成的新 token 数量
        block_size: 模型最大序列长度
        temperature: 控制随机性，越高越随机
        top_k: 只从概率最高的 k 个 token 中采样

    TODO:
    1. 循环 max_new_tokens 次:
       a. 截断 idx 到最后 block_size 个 token
       b. 前向传播得到 logits (B, T, vocab_size)
       c. 只取最后一个时间步的 logits: (B, vocab_size)
       d. 除以 temperature
       e. (可选) 如果 top_k 不为 None，把非 top_k 的设为 -inf
       f. softmax 得到概率分布
       g. 从分布中采样一个 token: torch.multinomial
       h. 拼接到 idx 后面
    2. 返回完整的 idx

    提示:
    - top_k 过滤: v, _ = torch.topk(logits, top_k)
                  logits[logits < v[:, [-1]]] = float('-inf')
    - 采样: torch.multinomial(probs, num_samples=1)
    """
    pass


# ============ 验证 ============
if __name__ == "__main__":
    # 这里用一个简单的假模型来验证你的 generate 函数逻辑
    class DummyModel(torch.nn.Module):
        """一个总是均匀随机输出的假模型"""
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, idx):
            B, T = idx.shape
            logits = torch.zeros(B, T, self.vocab_size)
            return logits, None

    vocab_size = 50
    block_size = 16
    model = DummyModel(vocab_size)
    model.eval()

    start_ids = torch.zeros((1, 1), dtype=torch.long)

    with torch.no_grad():
        result = generate(model, start_ids, max_new_tokens=20,
                         block_size=block_size, temperature=1.0)

    if result is not None:
        assert result.shape == (1, 21), f"形状应为 (1, 21)，得到 {result.shape}"
        assert result[0, 0] == 0, "第一个 token 应保持不变"
        print("generate 函数通过基本验证!")
        print(f"生成的 token ids: {result[0].tolist()}")
    else:
        print("generate 返回了 None，请实现该函数")
