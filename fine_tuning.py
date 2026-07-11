import torch
import torch.nn.functional as F
import tiktoken
from gpt import GPT


# ====== GPT-2 超参数 ======
vocab_size = 50257   # GPT-2 BPE 词表大小
n_embd = 768
n_head = 12
n_layer = 12
block_size = 1024


# ====== 微调超参数 ======
batch_size = 4          # GPT-2 很大，batch 要小
learning_rate = 1e-5    # 微调用小学习率，防止破坏预训练知识
max_steps = 500
eval_interval = 50
eval_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


# ====== 加载预训练权重 ======
def from_pretrained(model_type='gpt2'):
    from transformers import GPT2LMHeadModel

    # 创建空模型（超参数和 GPT-2 一致）
    model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.mask')]
    # ↑ 过滤掉 register_buffer 注册的 mask，它不是参数

    # 加载 HuggingFace 模型
    print(f"从 HuggingFace 加载 {model_type} 权重...")
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = [k for k in sd_hf.keys()
                if not k.endswith('.attn.masked_bias')
                and not k.endswith('.attn.bias')]

    # 键名已对齐，数量应该一致
    assert len(sd_keys) == len(sd_keys_hf), \
        f"参数数量不匹配: 你的模型 {len(sd_keys)}, HF {len(sd_keys_hf)}"

    # HF 的 Conv1D 权重需要转置的四个
    transposed = [
        'attn.c_attn.weight',
        'attn.c_proj.weight',
        'mlp.c_fc.weight',
        'mlp.c_proj.weight',
    ]

    for k in sd_keys_hf:
        if any(k.endswith(t) for t in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    model.load_state_dict(sd)
    print(f"成功加载 {len(sd_keys_hf)} 个参数")
    return model


# ====== Tiktoken 编码器 ======
enc = tiktoken.get_encoding("gpt2")


# ====== 数据加载 ======
def load_data(filepath="input.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = enc.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ====== 评估 ======
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    output = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for i in range(eval_steps):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            losses[i] = loss.item()
        output[split] = losses.mean().item()
    model.train()
    return output


# ====== 生成 ======
@torch.no_grad()
def generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    model.eval()
    tokens = enc.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # 截断到 block_size
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=-1)

    model.train()
    return enc.decode(idx[0].cpu().tolist())


# ====== 微调主循环 ======
def finetune():
    # 加载预训练模型
    model = from_pretrained('gpt2')
    model = model.to(device)

    # 先测试一下预训练效果
    print("\n--- 微调前生成 ---")
    print(generate(model, "To be or not to be,"))

    # 加载莎士比亚数据
    train_data, val_data = load_data("input.txt")
    print(f"\n训练 tokens: {len(train_data)}, 验证 tokens: {len(val_data)}")

    # 优化器：微调通常只用小学习率的 AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练
    model.train()
    for step in range(max_steps):
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {step:4d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

        x, y = get_batch('train', train_data, val_data)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 微调后生成
    print("\n--- 微调后生成 ---")
    print(generate(model, "To be or not to be,"))
    print(generate(model, "KING HENRY:"))


if __name__ == "__main__":
    finetune()