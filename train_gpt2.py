import torch
import torch.nn.functional as F
from gpt import GPT

# ====== 超参数 ======
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
filepath = "input.txt"


# ====== Tokenizer ======
with open(filepath,'r',encoding = 'utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {char:i for i, char in enumerate(chars)}
itos = {i:char for i, char in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode = lambda l:''.join([itos[i] for i in l])


# ====== 构造训练/验证数据 ======
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# ====== Batch 构造器 ======
def get_batch(split, train_data, val_data):
    """
    1. 根据 split 选择 train_data 或 val_data
    2. 随机选取 batch_size 个起始位置
    3. 构造 x (input) 和 y (target, 即 x 右移一位)
    4. 移动到 device
    返回: x (B, T), y (B, T)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


# ====== 评估函数 ======
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """
    1. 将模型设为 eval 模式
    2. 对 train 和 val 各跑 eval_steps 个 batch
    3. 计算平均 loss
    4. 将模型设回 train 模式
    返回: {"train": avg_train_loss, "eval": avg_val_loss}
    """
    model.eval()
    output = {}
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_steps)
        for i in range(eval_steps):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            losses[i] = loss.item()
        output[split] = losses.mean().item()
    model.train()
    return output


# ====== Generator ======
def Generator(model, idx, n_new_tokens, temperature = 1, topk = None):
    """
    利用GPT模型，实现输入序列到输出序列的自动转换
    输入：idx, 输入序列组成的列表;n_new_tokens数;温度
    输出：模型预测的当前上下文的下n_new_tokens个字符，拼接在原上下文后，形状(1, T + n_new_tokens)
    """
    idx = torch.tensor(encode(idx), dtype = torch.long, device = device).view(1, -1) # (1, len(idx))
    for _ in range(n_new_tokens):
        input = idx[:, -model.block_size:] # (1, block_size)
        logits, _ = model(input) # (1, block_size, vocab_size)
        last_logits = logits[:, -1, :] / temperature # (1, vocab_size)
        if topk is not None:
            v, _ = torch.topk(last_logits, topk)
            last_logits[last_logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(last_logits, dim = -1) # (1, vocab_size)
        next_token = torch.multinomial(probs, num_samples = 1) # (1, 1)
        idx = torch.cat((idx,next_token), dim = -1)
    
    text = decode(idx[0].cpu().tolist())
    print("序列为：")
    print(f"{text}")


# ======Training Loop ======
def train():
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    for step in range(max_steps):
        if step % eval_interval == 0 or step == max_steps - 1:
            output = estimate_loss(model, train_data, val_data)
            print(f"第{step:3d}步的平均训练损失为:{output['train']:.4f}", end = ',')
            print(f"第{step:3d}步的平均验证损失为:{output['eval']:.4f}")
        x, y = get_batch('train', train_data, val_data)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("\n训练完毕\n")
    Generator(model, 'sir,', 200)

if __name__ == "__main__":
    train()

