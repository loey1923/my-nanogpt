"""
练习 02 (续): 构造训练 batch
"""
import torch

# --- 假设你已完成 Part 1, 这里直接给出结果 ---
text = """To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles."""

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

BLOCK_SIZE = 8
BATCH_SIZE = 4

# --- Step 3: 把整个文本编码成 tensor ---
# TODO 4: 用 encode 编码整段文本，然后转成 torch.long tensor
data = ...  # torch.tensor(encode(text), dtype=torch.long)
print(f"data shape: {data.shape}, dtype: {data.dtype}")

# --- Step 4: 理解训练样本的构造 ---
# 一个训练样本：x = data[i:i+BLOCK_SIZE], y = data[i+1:i+BLOCK_SIZE+1]
# x 是输入, y 是对应的"下一个字符"标签
# 展示一个例子
x_example = data[:BLOCK_SIZE]
y_example = data[1:BLOCK_SIZE+1]
print(f"\n输入:  {decode(x_example.tolist())!r}")
print(f"标签:  {decode(y_example.tolist())!r}")
print("每一步的预测任务:")
for t in range(BLOCK_SIZE):
    context = x_example[:t+1]
    target = y_example[t]
    print(f"  {decode(context.tolist())!r:20s} -> '{itos[target.item()]}'")


# --- Step 5: 随机采样 batch ---
def get_batch(data, block_size, batch_size):
    """
    从 data 中随机采样一个 batch

    返回:
        x: shape (batch_size, block_size) 输入
        y: shape (batch_size, block_size) 标签
    """
    # TODO 5: 随机生成 batch_size 个起始位置
    # 提示: torch.randint(high, (size,))
    # high 应该是 len(data) - block_size
    ix = ...

    # TODO 6: 用这些起始位置从 data 中切出 x 和 y
    # 提示: torch.stack([data[i:i+block_size] for i in ix])
    x = ...
    y = ...
    return x, y


# 验证
xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
print(f"\n✓ batch shapes: x={xb.shape}, y={yb.shape}")
assert xb.shape == (BATCH_SIZE, BLOCK_SIZE)
assert yb.shape == (BATCH_SIZE, BLOCK_SIZE)
print("✓ 数据加载完成！")
