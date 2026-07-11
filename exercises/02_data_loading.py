"""
练习 02: Tokenizer 与数据加载
==============================
目标：把原始文本变成模型能吃的 tensor batch
这是 nanogpt 训练 pipeline 的第一步
"""
import torch

# --- 配置 ---
BLOCK_SIZE = 8      # 每个训练样本的序列长度 (context length)
BATCH_SIZE = 4      # 每个 batch 多少个样本

# --- Step 1: 读取数据 ---
# 我们用一段简单文本作为训练数据
text = """To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles."""

print(f"文本长度: {len(text)} 字符")

# --- Step 2: 字符级 Tokenizer ---
# TODO 1: 从文本中提取所有不重复字符并排序，得到 vocab
# 提示: sorted(list(set(text)))
chars = ...
vocab_size = len(chars)
print(f"词表大小: {vocab_size}")

# TODO 2: 创建 char->int 和 int->char 的映射字典
# stoi: string to integer, itos: integer to string
stoi = ...  # {ch: i for i, ch in enumerate(chars)}
itos = ...  # {i: ch for i, ch in enumerate(chars)}


# TODO 3: 实现 encode 和 decode 函数
def encode(s):
    """把字符串转成整数列表"""
    ...  # [stoi[c] for c in s]


def decode(l):
    """把整数列表转回字符串"""
    ...  # ''.join([itos[i] for i in l])


# 验证
assert decode(encode("hello")) == "hello", "encode/decode 不一致"
print(f"✓ 'hello' -> {encode('hello')} -> '{decode(encode('hello'))}'")
