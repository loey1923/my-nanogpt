"""
练习 7: 训练循环
把所有组件串起来，在真实数据上训练你的 GPT 模型。
"""
import torch
import torch.nn.functional as F

# 从你之前的练习中导入（或直接复制过来）
# from ex06_full_gpt_model import GPT

# ============ 超参数 ============
batch_size = 32
block_size = 64
n_embd = 64
n_head = 4
n_layer = 3
learning_rate = 3e-4
max_steps = 1000
eval_interval = 100
eval_steps = 20
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============ 数据准备 ============
def load_data(filepath="input.txt"):
    """
    TODO:
    1. 读取文本文件
    2. 建立字符级词表 (chars = sorted(list(set(text))))
    3. 创建 encode/decode 函数
    4. 将文本编码为 tensor
    5. 划分 train/val (90%/10%)
    返回: train_data, val_data, encode, decode, vocab_size
    """
    pass


# ============ 批次采样 ============
def get_batch(split, train_data, val_data):
    """
    TODO:
    1. 根据 split 选择 train_data 或 val_data
    2. 随机选取 batch_size 个起始位置
    3. 构造 x (input) 和 y (target, 即 x 右移一位)
    4. 移动到 device
    返回: x (B, T), y (B, T)
    """
    pass


# ============ 评估函数 ============
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """
    TODO:
    1. 将模型设为 eval 模式
    2. 对 train 和 val 各跑 eval_steps 个 batch
    3. 计算平均 loss
    4. 将模型设回 train 模式
    返回: {"train": avg_train_loss, "val": avg_val_loss}

    提示: 用 model.eval() / model.train() 切换模式
         用 torch.no_grad() 避免计算梯度（已用装饰器）
    """
    pass


# ============ 主训练循环 ============
def train():
    """
    TODO:
    1. 加载数据
    2. 创建模型，移动到 device
    3. 创建 AdamW optimizer
    4. 循环 max_steps 步:
       a. 每 eval_interval 步评估一次并打印 loss
       b. 采样一个 batch
       c. 前向传播得到 logits
       d. 计算 cross_entropy loss
       e. 反向传播: optimizer.zero_grad() → loss.backward() → optimizer.step()
    5. 训练完毕后，生成一段文本展示效果

    提示: cross_entropy 需要 reshape:
         loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    """
    pass


if __name__ == "__main__":
    train()
