"""03_single_head_attention.py 的验证部分 - 接在主文件末尾运行"""
# (将此内容追加到 03_single_head_attention.py 末尾，或者直接运行此文件)

# 把上面的文件 import 进来也行，这里为了方便直接放验证逻辑
# 假设你已经完成了 TODO，下面是验证代码

if __name__ == "__main__":
    # 验证
    print("=" * 50)
    print("验证单头自注意力")
    print("=" * 50)

    attn = SingleHeadAttention(N_EMBD, HEAD_SIZE, BLOCK_SIZE)

    # 随机输入
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBD)
    out = attn(x)

    # 检查输出形状
    expected_shape = (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE)
    assert out.shape == expected_shape, \
        f"输出形状错误: {out.shape}, 期望 {expected_shape}"
    print(f"[PASS] 输出形状正确: {out.shape}")

    # 验证因果性: 改变位置 5 的输入不应影响位置 0-4 的输出
    x2 = x.clone()
    x2[:, 5, :] = torch.randn(BATCH_SIZE, N_EMBD)
    out2 = attn(x2)

    # 位置 0-4 应该完全不变
    assert torch.allclose(out[:, :5, :], out2[:, :5, :]), \
        "因果性验证失败! 修改位置5影响了位置0-4的输出"
    print("[PASS] 因果掩码正确: 未来位置的修改不影响过去")

    # 位置 5 之后应该不同
    assert not torch.allclose(out[:, 5:, :], out2[:, 5:, :]), \
        "修改位置5后，位置5及之后应该变化"
    print("[PASS] 位置5之后的输出正确发生了变化")

    print("\n所有验证通过! 你已经实现了 self-attention 的核心!")
    print("这就是 'Attention is All You Need' 论文的核心公式:")
    print("Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V")
