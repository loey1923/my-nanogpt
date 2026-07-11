# NanoGPT 手写练习

按顺序完成以下练习，每个文件中的 `TODO` 是你需要填写的部分。
完成后运行 `python <文件名>` 即可验证。

## 顺序

1. `01_tensor_basics.py` — PyTorch 张量操作与 nn.Module 基础
2. `02_data_loading.py` — 字符级 tokenizer + batch 构造
3. `03_single_head_attention.py` — 单头 self-attention
4. `04_multi_head_attention.py` — 多头 attention
5. `05_transformer_block.py` — Transformer block（attention + FFN + residual + LN）
6. `06_gpt_model.py` — 完整 GPT 模型
7. `07_training_loop.py` — 训练循环
8. `08_generate.py` — 自回归文本生成

## 数据

练习使用 Shakespeare 文本。首次运行 02 会自动下载。

## 提示

- 卡住时回去看 `train_gpt2.py`，但尽量先自己写
- 每步都有 shape 断言帮你检查，跑通就说明写对了
- 用 CPU 就行，不需要 GPU
