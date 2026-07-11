# **动手实现built-nanogpt项目**

## 写在前面

### 学习路线图

当前对transformer建立了模糊理解，开始着手实现nanogpt以厘清细节，并初步掌握Pytorch库。
完成项目的路径为：

- 已有: 线性代数 + 微积分
- 需要建立: 统计学习基础（loss函数、概率模型的直觉）
- 需要建立: 神经网络基础（前向传播、反向传播、梯度下降）
- 需要建立: 深度学习组件（Embedding、归一化、激活函数）
- 目标: Transformer 架构（Attention机制、GPT结构）
- 最终: 读懂并推演 nanogpt 代码

### nanogpt项目的提交记录分层

<details>
    <summary>点击展开/收起</summary>

**阶段 1：最小可运行原型（核心架构）**
从空文件到能跑通一次前向传播和生成。

| commit | 内容 |
|---|---|
| `28916d9` | 初始提交，空文件 |
| `dd03c64` | 写 GPT2 的 `nn.Module` 和 `forward()` |
| `af6e591` | 实现从模型生成文本 |
| `ce84eb3` | 自动检测设备（CPU/GPU） |
| `92b5bf9` | 加入莎士比亚数据，构造第一个 batch |
| `41078d1` | 计算 loss（交叉熵） |
| `7822fce` | 跑通一个简单训练循环 |

**阶段 2：数据加载与基础训练设施**

| commit | 内容 |
|---|---|
| `631f7d6` | 写 `DataLoaderLite`，管理 batch 读取 |
| `ec39245` | 权重绑定（embedding 与 unembedding 矩阵共享） |
| `9ac321e` | GPT-2 权重初始化方式 |

**阶段 3：性能优化（精度与编译）**

| commit | 内容 |
|---|---|
| `5265b20` | 开启 TF32 矩阵运算 |
| `177e4cd` | 切换到 bfloat16 混合精度 |
| `fb8bd6e` | 加入 `torch.compile` |
| `7ee630c` | 切换到 Flash Attention |
| `7230096` | vocab size 从 50257 扩展到 50304（对齐硬件友好的数字） |

**阶段 4：训练优化器与调度器**

| commit | 内容 |
|---|---|
| `5215868` | 美化打印输出 |
| `105f117` | AdamW 参数设置 + 梯度裁剪 |
| `90e5d15` | 学习率调度器（cosine decay） |
| `3a148e4` | weight decay 只作用于 2D 参数 + fused AdamW |
| `01be6b3` | 梯度累积（gradient accumulation） |

**阶段 5：分布式训练与评测（了解即可，非架构核心）**

| commit | 内容 |
|---|---|
| `ba2554a` | DDP（多卡分布式训练） |
| `69cb21f` | 切换到 FineWeb-EDU 数据集 |
| `21d3d32` | 加入 validation split |
| `8018ed2` | 加入 HellaSwag 评测 |
| `efedfac` | 加入 checkpoint 保存 |

</details>

## **3.28**

### 阶段学习目标

1. 读懂Python中的`__call__`与运算符重载，掌握神经网络基础概念
2. 学习[micrograd](https://github.com/karpathy/micrograd)项目 + 跟着手写代码
3. 进入 build-nanogpt

### `__call__`与运算符重载

- `__call__`:一种“魔术方法”，使得**对象可以像函数一样被调用**，如定义了`__call__`的对象`obj`，`obj(x)`等价于`obj.__call__(x)`
- 运算符重载：通过定义“魔术方法”，可以**自定义对象的内置运算符**，如+-*/等的行为
  
## **3.29**

了解神经网络基础概念

### 神经网络基础概念

<details>
<summary>点击展开/收起</summary>

#### 层级一：单个神经元的运作

##### Weight 和 Bias

从一个最简单的情形开始。

假设你有一个神经元，接收两个输入 $x_1, x_2$，它做的事情是：

$$z = w_1 x_1 + w_2 x_2 + b$$

$w_1, w_2$ 是 weight， $b$ 是 bias。

**weight 的含义**：对输入的重视程度. $w_1$ 越大， $x_1$ 对结果的影响越大。

**bias 的含义**：一个与输入无关的偏移量。它的作用是让神经元在输入全为0时也能输出非零值，相当于调整激活的「门槛」。

用矩阵写法，如果有 $n$ 个输入，这个神经元就是：

$$z = \mathbf{w}^T \mathbf{x} + b$$

这是一个线性变换。

---

##### 为什么需要激活函数

考虑两层神经元叠加，没有激活函数：

$$z_2 = W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$$

两个矩阵相乘还是一个矩阵，两层叠加等价于一层。无论叠多少层，表达能力和单层完全相同，网络的深度没有意义。

激活函数 $f$ 插入两层之间：

$$z_2 = W_2 \cdot f(W_1 x + b_1) + b_2$$

因为 $f$ 是非线性的， $W_2 \cdot f(\cdot)$ 不能再被化简为单个矩阵乘法，深度从此有意义。

**激活函数的本质定义**：施加在线性变换结果上的非线性函数。输出范围是什么不重要，非线性是核心。

常见的几个：
- Sigmoid：输出 $(0,1)$，早期常用，现在主干网络很少用
- ReLU： $\max(0, x)$, 负数直接归零，计算极快
- GELU：比 ReLU 在负区间有小幅非零输出，Transformer 的标准选择

---

##### 前向传播（Forward Pass）

前向传播就是：**给定输入，按层顺序依次计算，直到得到输出**。

以一个两层网络为例：

```
输入 x
  ↓
z1 = W1·x + b1        # 第一层线性变换
  ↓
a1 = ReLU(z1)         # 第一层激活
  ↓
z2 = W2·a1 + b2       # 第二层线性变换
  ↓
输出 z2
```

「前向」指的是数据从输入向输出方向流动。这个方向与后面反向传播的「反向」相对。

---

#### 层级二：网络如何学习

##### Loss 函数

网络有了输出之后，需要一个方式衡量「输出距离正确答案有多远」。这个衡量值就是 loss（也叫损失或代价）。

Loss 必须满足一个条件：**是一个标量**（单个数值），这样才能对参数求导。

以语言模型为例，网络输出的是下一个 token 是词表中每个词的概率分布，正确答案是某个具体的 token。这里用的 loss 是**交叉熵**：

$$L = -\log P(\text{正确token})$$

直觉：如果网络对正确答案给出的概率很高， $\log P$ 接近0，loss 小；如果概率很低， $\log P$ 是一个很大的负数，loss 大。

---

##### 梯度

Loss 是所有参数（weights 和 biases）的函数。梯度是：

$$\frac{\partial L}{\partial w_i}$$

即 loss 对每个参数的偏导数。

**它的含义**：如果把参数 $w_i$ 增大一点点，loss 会增大还是减小，以及变化的幅度。

梯度是一个与参数形状完全相同的张量，每个位置存的是 loss 对该位置参数的偏导数。

---

##### 计算图

前向传播的每一步操作都被 PyTorch 记录成一张有向图，节点是中间结果，边是操作。

以 $z = w \cdot x + b$ 为例：

```
w ──┐
    multiply ──→ (w·x) ──┐
x ──┘                    add ──→ z
                    b ───┘
```

这张图的作用是：**反向传播时，沿着图的反方向，用链式法则逐步计算每个节点对 loss 的梯度。**

PyTorch 在你每次做张量运算时自动建立这张图，`loss.backward()` 就是触发它的入口。

---

##### 反向传播（Backpropagation）

反向传播是链式法则在计算图上的机械展开。

假设计算过程是 $L = f(g(w))$，链式法则给出：

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial w}$$

在计算图上，这个过程从 loss 节点出发，沿边反向传播，每经过一个操作节点就乘以该操作的局部梯度，最终到达每个参数节点，积累出该参数的完整梯度。

**关键结论**：你不需要手动对整个网络求导。只要每个基本操作（加法、乘法、激活函数等）知道自己的局部梯度公式，反向传播就能自动组合出任意深度网络的梯度。这是 PyTorch 的 Autograd 系统做的事。

---

##### 梯度下降

有了每个参数的梯度，更新规则是：

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

$\eta$ 是学习率（learning rate），控制每步更新的幅度。

**为什么是减去梯度**：梯度指向 loss 增大最快的方向，减去它就是向 loss 减小最快的方向走一步。

反复执行「前向传播计算loss → 反向传播计算梯度 → 梯度下降更新参数」这个循环，网络的输出会逐渐向正确答案靠近。

---

#### 层级三：工程组织方式

##### Batch

不把单个样本逐个送入网络，而是把多个样本打包成一个张量同时计算。

原因有两个：
- **效率**：GPU 擅长大规模并行计算，一次处理256个样本比处理256次单个样本快得多
- **梯度稳定性**：单个样本的 loss 噪声很大，多个样本平均后梯度方向更稳定

Batch size 记为 $B$。如果单个样本形状是 $(T, C)$，打包后就是 $(B, T, C)$。

---

##### Epoch 和 Iteration

- **iteration（步）**：用一个 batch 做一次「前向+反向+更新」的完整循环
- **epoch（轮）**：把整个训练数据集完整过一遍所需的 iteration 数

如果训练集有10000个样本，batch size 是100，那么一个 epoch = 100 iterations。

---

##### Tensor 的维度语义

nanogpt 里最核心的张量形状是 $(B, T, C)$：

- $B$（Batch size）：一次处理多少个独立序列
- $T$（Time / Sequence length）：每个序列有多少个 token
- $C$（Channel / n\_embd）：每个 token 用多少维的向量表示

这三个字母是 Transformer 代码里的惯用命名，你在源码里会反复看到它们。

---

##### 广播（Broadcasting）

形状不完全相同的张量之间做运算时，PyTorch 会自动沿某些维度「复制扩展」较小的张量，使双方形状匹配。

规则是从右向左对齐维度，大小为1的维度可以被扩展：

```python
# 形状 (B, T, C) 和 (1, T, C) 相加
# (1, T, C) 会被自动扩展成 (B, T, C)
# 不发生实际的内存复制，只是计算时重复使用
```

在 Transformer 里，position embedding `(T, C)` 加到 token embedding `(B, T, C)` 上，就依赖这个规则。

---

这三个层级的概念之间的完整依赖链是：

```
weight/bias 定义线性变换
    ↓
激活函数使叠加有意义
    ↓
前向传播产生输出
    ↓
loss 函数量化误差
    ↓
计算图记录运算路径
    ↓
反向传播沿图计算梯度
    ↓
梯度下降更新参数
    ↓
batch 把以上过程并行化
    ↓
(B,T,C) 张量语义 + 广播 是并行化的数学表达方式
```

</details>

## **4.6**

通过[micrograd](https://github.com/karpathy/micrograd)项目学习计算图与反向传播

### micrograd

<details>
    <summary>点击展开/收起</summary>

`engine.py`中定义的`Value`类是 Andrej Karpathy 编写的著名的 **micrograd**（微型自动求导引擎）的核心部分。它通过非常精简的代码实现了深度学习中最重要的概念：**计算图**与**反向传播（Backpropagation）**。

下面分两部分详细拆解：

---

#### 第一部分：从数学层面理解反向传播

##### 1. 什么是反向传播？
反向传播本质上是微积分中 **链式法则（Chain Rule）** 在计算图（有向无环图）上的系统性应用。
在机器学习中，我们通常有一个复杂的函数（比如神经网络），输入是数据和权重，输出是损失值（Loss）。我们想要知道：**每一个权重发生微小变化时，会对最终的损失值产生多大的影响？** 这个“影响的比例”就是**梯度（Gradient）**。

链式法则告诉我们，如果 $y = f(u)$ 且 $u = g(x)$，那么 $y$ 对 $x$ 的导数为：
$$ \frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x} $$

**大白话解释**：
假设你想买一辆车（最终目标 $L$）。车的价格取决于铁矿石的价格（节点 $q$）。而铁矿石的价格又取决于挖掘机的租金（节点 $x$）。
如果挖掘机租金涨了 1 块钱，铁矿石涨 2 块钱（局部导数 = 2）；铁矿石涨 1 块钱，车价涨 10 块钱（全局传回来的导数 = 10）。
那么挖掘机租金涨 1 块钱，车价就会涨 $2 \times 10 = 20$ 块钱。这就是反向传播的过程：**从最终目标开始，一步步往前乘局部变化率**。

##### 2. 举例说明
假设我们有这样一个数学算式，并设定初始值 $x=2, y=-3, z=10$：
$$ L = (x \cdot y + z)^2 $$

**步骤1：前向传播（Forward Pass）——计算结果**
我们将算式拆解为基本运算（对应代码中的 `Value` 节点）：

1. $q = x \cdot y = 2 \times (-3) = -6$
2. $p = q + z = -6 + 10 = 4$
3. $L = p^2 = 4^2 = 16$

**步骤2：反向传播（Backward Pass）——计算梯度**
我们的目标是求 $\frac{\partial L}{\partial x}, \frac{\partial L}{\partial y}, \frac{\partial L}{\partial z}$。我们从后往前算：

1. **从 L 自身开始**：
   $$ \frac{\partial L}{\partial L} = 1 $$ （对应代码中 `self.grad = 1`）

2. **经过平方运算求 $p$ 的梯度**（因为 $L = p^2$）：
   局部导数 $\frac{\partial L}{\partial p} = 2p = 2 \times 4 = 8$
   根据链式法则：$\frac{\partial L}{\partial p} = \frac{\partial L}{\partial p} \times \frac{\partial L}{\partial L} = 8 \times 1 = 8$

3. **经过加法运算求 $q$ 和 $z$ 的梯度**（因为 $p = q + z$）：
   加法的局部导数都是 1（即 $\frac{\partial p}{\partial q}=1, \frac{\partial p}{\partial z}=1$）。
   根据链式法则，加法相当于把梯度 **原封不动地分配（路由）** 给两个输入：
   $$ \frac{\partial L}{\partial q} = 1 \times \frac{\partial L}{\partial p} = 1 \times 8 = 8 $$
   $$ \frac{\partial L}{\partial z} = 1 \times \frac{\partial L}{\partial p} = 1 \times 8 = 8 $$

4. **经过乘法运算求 $x$ 和 $y$ 的梯度**（因为 $q = x \cdot y$）：
   乘法的局部导数是互相交换的（即 $\frac{\partial q}{\partial x} = y, \frac{\partial q}{\partial y} = x$）。
   $$ \frac{\partial L}{\partial x} = y \times \frac{\partial L}{\partial q} = -3 \times 8 = -24 $$
   $$ \frac{\partial L}{\partial y} = x \times \frac{\partial L}{\partial q} = 2 \times 8 = 16 $$

至此，我们就求出了所有输入的梯度！

---

#### 第二部分：从代码层面分析如何实现计算图与反向传播

这段代码巧妙地将“数值计算”和“计算图构建”结合在了一起。我们逐块分析：

##### 1. 核心数据结构：`Value` 节点
每个 `Value` 实例既代表图中的一个**节点**，也记录了前向和后向的数据。
*   `self.data`: 保存前向传播的计算结果（如上面的 16, 4, -6）。
*   `self.grad`: 保存反向传播计算出的梯度（初始为 0）。
*   `self._prev`: 保存生成这个节点的“父节点”（也就是输入的参数）。这就在代码层面把独立的节点**连接成了有向无环图（DAG）**。
*   `self._backward`: 一个闭包函数（函数内部的函数），它知道**针对当前这个特定的运算符，应该如何应用链式法则计算输入节点的梯度**。

##### 2. 前向传播与动态建图（以乘法 `__mul__` 为例）
当我们执行 `c = a * b` 时，Python 会调用 `a.__mul__(b)`：
```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other) # 包装常量
    out = Value(self.data * other.data, (self, other), '*')     # 1. 计算前向结果，2. 建立图连接 (self, other)

    def _backward():
        # 核心：链式法则 = 局部导数 * 全局传回的导数(out.grad)
        # 如果 out = self * other
        # 那么 d(out)/d(self) = other.data
        # d(out)/d(other) = self.data
        self.grad += other.data * out.grad 
        other.grad += self.data * out.grad
    out._backward = _backward # 将如何求导的逻辑挂载到输出节点上

    return out
```
**注意为什么要用 `+=` 累加梯度？**
因为在复杂的计算图中，一个变量可能被多处使用（比如 $y = x \cdot x$ 或者分叉到图的多个分支）。根据微元法中的多元链式法则，它的总梯度应该是所有使用了它的路径回传的**梯度之和**。

##### 3. 反向传播引擎：`backward()` 函数
当我们对最终结果（比如 Loss）调用 `L.backward()` 时，引擎开始运转。这里有两步：

**第一步：拓扑排序（Topological Sort）**

```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child) # 递归找父节点
        topo.append(v)#子节点都遍历插入后再插入父节点，后序遍历
build_topo(self)
```
**为什么需要拓扑排序？**
在反向传播时，如果我们要计算节点 A 的梯度，我们必须**先确保所有依赖 A 的节点的梯度都已经计算完毕**。拓扑排序保证了所有节点的排列顺序是：**被依赖者在前，依赖者在后**。
经过上面的递归调用后，`topo` 列表里存放的是从“输入起点”到“最终输出”的顺序。

**第二步：反向链式调用**
```python
self.grad = 1 # 将最终输出对自己的导数设为 1（启动器）
for v in reversed(topo): # 反转拓扑顺序，从后往前（从 Loss 向 Input）计算
    v._backward()        # 触发前向传播时预定义的局部求导公式
```
遍历 `reversed(topo)` 意味着从最终输出节点开始，依次执行每个节点的 `_backward()` 函数，梯度就这样沿着 `_prev` 建立的图结构，像流水一样倒流（Backpropagate）回去了。

##### 4. 代码的工程巧思（魔法方法）
你会看到 `__sub__`, `__truediv__` 等方法并没有写复杂的 `_backward` 逻辑，而是直接复用了已有的算子：
*   减法：`self - other` 被转化为 `self + (-other)`
*   除法：`self / other` 被转化为 `self * (other ** -1)`
这种设计极大地简化了代码，只要保证加法（`+`）、乘法（`*`）和幂运算（`**`）的导数逻辑正确，减法和除法的求导就会自然而然地通过现成的计算图得到正确结果。

#### 总结
这段代码的美妙之处在于：当我们用普通的 Python 语法写数学公式计算 Loss 时（前向传播），底层实际上在**悄悄地牵线搭桥（记录 `_prev`）并塞入说明书（记录 `_backward` 闭包）**，构建了一张庞大的有向计算图。当最后大喊一声 `backward()` 时，程序就沿着这张图逆流而上，把梯度完美地分配给了每一个初始变量。
    
</details>

## **4.19**

### 代码基础
掌握[简单Numpy、Pytorch、Pandas基础](https://htmlpreview.github.io/?https://github.com/loey1923/my-nanogpt/blob/main/numpy_pytorch_pandas_guide.html)

## **7.5 & 7.6 & 7.7**

### 阅读build-nanogpt源码

<details>
    <summary>点击展开/收起</summary>

#### build-nanogpt 项目总结

这是 Karpathy 的教学项目，用 **44 个 commit、从空文件到完整复现 GPT-2 124M**，约 1 小时约 $10 跑出能说人话的语言模型。它的价值不在"又造了一个 GPT-2"，而在把**现代深度学习训练系统的每个组件**拆成可独立理解的一步步。

---

#### 七阶段演进路线

```
空文件 ──► 能加载预训练权重生成 ──► 能从零训练 ──► 训得稳且快
       ──► 优化器精细配置 ──► 模拟大batch/多卡 ──► 真实数据+验证 ──► 评测+存档+看板
```

| 阶段              | Commits | 核心突破                                                     | 验证标志                                          |
| ----------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------- |
| **1 架构**        | 1–3     | 定义 GPT/Block/Attention/MLP，对齐 HF 权重                   | `didn't crash yay!`，生成连贯英文                 |
| **2 训练循环**    | 4–9     | device检测、Shakespeare数据、cross entropy、4步训练循环、weight tying | 单batch loss 11→0.003（过拟合验证）               |
| **3 初始化+精度** | 10–13   | GPT-2残差缩放初始化、TF32、bf16、torch.compile               | 初始loss≈10.95（接近理论10.82），速度~10000 tok/s |
| **4 优化器**      | 14–17   | vocab 50304、AdamW精细参数、grad clip、cosine+warmup、weight decay分组、fused | 训练稳定不发散                                    |
| **5 大batch**     | 18–20   | 梯度累积（单卡模拟大batch）、DDP（多卡并行）、梯度同步优化   | 8卡×4累积=32 micro，全局50万token batch           |
| **6 真实数据**    | 21–23   | FineWeb-Edu 10B、shard分片流式加载、uint16存储、验证集       | val loss 诚实反映泛化                             |
| **7 评测存档**    | 24–29   | HellaSwag标准化评测、checkpoint、日志、采样生成、可视化      | HellaSwag 0.25→0.30+，完整训练闭环                |

---

#### 每个阶段解决的核心矛盾

| 阶段 | 矛盾                         | 解法                                       | 贯穿思想                                 |
| ---- | ---------------------------- | ------------------------------------------ | ---------------------------------------- |
| 1    | "我懂 transformer 但没写过"  | 对齐 HF 权重验证架构正确                   | **用预训练权重做单元测试**               |
| 2    | "架构对但不会训练"           | 4步循环 + 过拟合单batch                    | **先让它记住128个token，证明pipeline通** |
| 3    | "能训但慢且不稳"             | 残差缩放 + TF32/bf16 + compile             | **精度与速度的权衡**                     |
| 4    | "默认优化器不适合大模型"     | betas调小 + grad clip + cosine + decay分组 | **为大模型训练专门调参**                 |
| 5    | "显存装不下50万token的batch" | 梯度累积 + DDP 叠加                        | **用时间换空间，用通信换算力**           |
| 6    | "Shakespeare是玩具"          | FineWeb-Edu + 验证集                       | **从背课文到真学习**                     |
| 7    | "loss不等于能力"             | HellaSwag + checkpoint + 可视化            | **标准化评测 + 可恢复 + 可观测**         |

---

#### 关键技术点（按重要性）

**架构层**

- `c_attn` 合并 QKV 一次 matmul，下三角 mask 注册为 buffer
- pre-LN 残差 `x + sublayer(ln(x))`，`GELU(tanh)` 对齐 GPT-2
- weight tying：嵌入矩阵和输出矩阵共享，省 3860 万参数

**初始化层**

- 权重 `N(0, 0.02²)`，残差路径 `c_proj` 额外乘 `(2L)^-0.5` 防方差爆炸
- 这是深层 transformer 训练稳定的**关键**，比任何优化器技巧都重要

**精度层**

- TF32：自动加速 matmul，Ampere+ GPU
- bf16 autocast：显存减半 + 速度提升，softmax/loss 保留 fp32
- `torch.compile`：JIT 融合 kernel（Windows 上常失败，Linux 上加速 20-50%）

**优化器层**

- `betas=(0.9, 0.95)`：LM 梯度噪声大，二阶动量衰减要快
- grad clip `max_norm=1.0`：安全阀防梯度爆炸
- cosine + warmup：前期别乱冲，后期精细收敛
- weight decay 只施加于 2D 参数：bias/LayerNorm 不 decay

**batch 层**

- 梯度累积：`loss /= grad_accum` 保证累加后是均值
- DDP：`require_backward_grad_sync` 只在最后一次反向同步，省通信
- 全局 batch = `micro × accum × world_size`

**数据层**

- shard 分片流式加载：装不下就分批
- uint16 存储：词表 < 2¹⁶，省一半磁盘
- 验证集独立：诚实反映泛化

**评测层**

- HellaSwag：把选择题转成"算各选项补全概率"，`acc_norm` 归一化长度
- checkpoint 存 `raw_model.state_dict()`：避开 DDP 前缀污染

---

#### 训练循环的最终形态

整个项目的核心就是这个循环，七阶段都在围绕它做加法：

```python
for step in range(max_steps):
    # 1. 定期评测（val loss / HellaSwag / 采样生成）
    if step % 250 == 0: evaluate(...)
    # 2. 清梯度
    optimizer.zero_grad()
    # 3. 梯度累积：N次 micro batch 串行前向+反向
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        with autocast(bf16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps       # 保证累加是均值
        loss.backward()                # 梯度自动累加到 param.grad
    # 4. 梯度裁剪（安全阀）
    clip_grad_norm_(model.parameters(), 1.0)
    # 5. 按余弦调度设学习率
    lr = get_lr(step); for g in optimizer.param_groups: g['lr'] = lr
    # 6. 更新参数（读 param.grad）
    optimizer.step()
    # 7. 记日志
    log(f"step {step} loss {loss}")
    # 8. 定期存档
    if step % 5000 == 0: torch.save(checkpoint, ...)
```

---

#### 学到的方法论

1. **逐步验证，不要一次写完**：先过拟合单 batch 确认 pipeline，再上真数据。每加一层复杂度都确认前一层没坏。
2. **用预训练权重做单元测试**：从 HF 加载权重跑前向，输出和 HF 一致就证明架构对——这比写测试用例高效。
3. **先正确再优化**：阶段 2 用最简单的训练循环跑通，阶段 3-4 才加 TF32/bf16/compile/调度。先保证正确，再加速。
4. **精度与速度的权衡贯穿始终**：TF32/bf16 牺牲一点精度换大幅加速；compile 牺牲灵活性换速度；禁用 compile 换回 HellaSwag 评测能力。
5. **观察比跑通更重要**：val loss、HellaSwag、采样生成、可视化——四个维度的"仪表盘"让你**看见**模型在学什么，而不是盲目跑数字。
   

</details>

## **7.8 & 7.9**

### 完成exercises\下的练习

由opus4.6阅读build-nangpt源码，将其拆解为[exercises\\](./exercises)下若干练习题。完成练习，熟练使用torch库函数与类，实现以下组件：

- 字符级Tokenizer：decode()/encode()，实现字符与token id之间的转换
- 单头掩码Self-Attention
- 多头掩码Self-Attention：由一个线性层同时计算q,k,v,由`.split()`拆分，再经过`.view()`与`.transpose()`转换为(B, n_head, T, head_size)形状的的多头形式，随后计算注意力、拼接多头、输出投影
- TransformerBlock:：LayerNorm → Multi-Head Attention → 残差连接 → LayerNorm → FFN (一层线性层+GeLU激活+一层线性层) → 残差连接
- GPT模型：tok_emb + pos_emb → N×TransformerBlock → LN_fin → lm_head，CrossEntropyLoss 计算，自回归 generate()
- Batch构造器：`get_batch()`随机采样，返回input与target(input右移一位)，用于自监督学习
- 训练循环逻辑：将所有组件组合，在训练数据上训练(前向传播、计算损失、反向传播、梯度下降优化)
- 文本生成器：自回归 + temperature 缩放 + top-k 过滤 + multinomial 采样

## **7.10**

### 完成模型定义与训练pipeline

完成[gpt.py](./gpt.py)中的`GPT`类定义与[train_gpt2](./train_gpt2.py)中的training loop，踩坑点：

- 输入张量的T指的是序列长度，即输入序列的token数，Multi-Head Attention的block_size指的是最大上下文长度，T不可超过block_size
- nn.embedding实际相当于一个查找表，例如

```python
......
self.token_embedding = nn.Embedding(vocab_size, n_embd)#输入token的种类最多有vocab_size种
self.position_embedding = nn.Embedding(block_size, n_embd)#输入token的位置最多有block_size个
......
B, T = idx.shape
position = torch.arange(T, device = idx.device)
x = self.position_embedding(position) + self.token_embedding(idx) # (B, T, n_embd)
......
```

其中`position_embedding()`是拿每一个`position`(形状为(T,))中的值在共有block_size项的表中查找对应项，替换为该项对应的n_embd维特征张量，位置嵌入的输出形状为(T, n_embd)，随后与词嵌入的输出相加，广播为(B, T, n_embd)

- `F.cross_entrophy()`输入的logits与targets，形状应变换为(B * T,  vocab_size)，(B * T, 1)
- `Generator`中取logits最后一个时间步，即对输入上下文的下一次的预测

## **7.11**

### 对齐 GPT-2 的结构，在预训练基础上微调

1. 修改`GPT` class 中定义的参数名，与 GPT-2 对齐

2. 在[fine_tune.py](./fine_tune.py)中，从 HuggingFace 加载 GPT-2 权重，换用 GPT-2 的 BPE Tokenizer

3. 类似 training loop，对预训练模型在莎士比亚训练集上做微调，前后对比：

   ```bash
   --- 微调前生成 ---
   To be or not to be, I'm not here to lecture anyone on how to make the most of your time. I've been here a couple of times, trying to become a better professional in a short time, and it's not going to ever be easy.
   
   But one thing I learned about myself is that your time is a resource.
   
   Here you find the source of your attention.
   
   It's what you put in the water in the morning, that makes a difference in your life.
   
   It's your body's response to your attention.
   
   Think about this.
   
   Your brain knows that your body doesn't feel the need to spend the time, energy and energy you put into it to respond to you.
   
   If you've never felt that way before, then you've probably thought about this one a little too much.
   
   You've probably thought about that one a little too much.
   
   You've probably thought about that one a little too much.
   
   You
   
   --- 微调后生成 ---
   To be or not to be, no man is a slave.
   
   KING RICHARD III:
   And why should you say you must not be?
   
   GREY:
   I do say there are some people who have
   like to be slaves to all: and I'll be it, King Edward.
   
   KING RICHARD III:
   No, that is not the case.
   
   GREY:
   When I was old, he would have the power, and I say
   not him. We were bound by the law of our time.
   
   KING RICHARD III:
   You know this, sir?
   
   GREY:
   I did in law.
   
   KING RICHARD III:
   You know that law?
   
   GREY:
   I do, sir.
   
   KING RICHARD III:
   And you know it, sir?
   
   GREY:
   I do, I know it.
   
   KING RICHARD III:
   I would not
   ```

### 项目总结

#### Karpathy 的 build-nanogpt 原项目

Karpathy 的项目是一个教学性质的 GPT-2 复现：用约 300 行 PyTorch 代码重新实现 GPT-2 的模型结构，加载 HuggingFace 的预训练权重验证正确性，然后在 FineWeb 数据集上从头训练一个 124M 参数的模型。项目展示了从模型定义、数据流水线、分布式训练到评估（HellaSwag）的完整流程。

---

#### 手写的部分：从零构建 GPT

##### 整体 Pipeline

```
原始文本 → Tokenizer(encode) → 整数序列 → 随机切片成batch → 模型前向 → logits → cross_entropy loss → backward → optimizer.step
```

这是所有语言模型训练的骨架，从头实现了每一环。

##### 模型结构（从外到内）

```
GPT
├── token_embedding: nn.Embedding(vocab_size, n_embd)
├── position_embedding: nn.Embedding(block_size, n_embd)
├── blocks: N × TransformerBlock
│   ├── ln_1 → MultiHeadAttention → 残差连接
│   └── ln_2 → FFN → 残差连接
├── ln_f: 最终 LayerNorm
└── lm_head: nn.Linear(n_embd, vocab_size, bias=False)
    └── weight tying: token_embedding.weight = lm_head.weight
```

##### 关键张量形状流

| 位置 | 张量 | 形状 |
|------|------|------|
| 输入 token ids | idx | (B, T) |
| token embedding | tok_emb | (B, T, n_embd) |
| position embedding | pos_emb | (T, n_embd) → 广播到 (B, T, n_embd) |
| 进入 attention 前 | x | (B, T, n_embd) |
| Q, K, V（合并计算） | c_attn(x) | (B, T, 3×n_embd) |
| 拆分后 Q/K/V | 各自 | (B, T, n_embd) |
| reshape 成多头 | q, k, v | (B, n_head, T, head_size) |
| attention score | q @ k^T | (B, n_head, T, T) |
| 因果掩码后 | masked_fill(-inf) | (B, n_head, T, T) |
| attention weight | softmax | (B, n_head, T, T) |
| attention 输出 | weight @ v | (B, n_head, T, head_size) |
| 拼接回去 | transpose+view | (B, T, n_embd) |
| FFN 中间层 | c_fc(x) | (B, T, 4×n_embd) |
| FFN 输出 | c_proj(gelu(x)) | (B, T, n_embd) |
| 最终 logits | lm_head(x) | (B, T, vocab_size) |
| loss 计算时 reshape | logits → (B×T, vocab_size), targets → (B×T,) | |

##### 掌握的核心机制

**Self-Attention 的本质：**

- Q 问"我在找什么"，K 答"我有什么"，V 是"我的内容"
- score = Q @ K^T / √d_k — 缩放防止梯度消失在 softmax 的饱和区
- 因果掩码：下三角矩阵，位置 i 只能看到 ≤i，masked_fill(-inf) 后 softmax 变成 0

**多头的意义：**

- 一个大空间拆成多个子空间并行关注不同模式
- 实现上：一次大矩阵乘法 → reshape → 分头计算 → 拼回来

**Pre-Norm 残差连接：**

```python
x = x + sublayer(layernorm(x))  # 不是 layernorm(x + sublayer(x))
```

残差加的是进入子层之前的原始值，保证梯度直通。

**Weight Tying：** embedding 矩阵和 lm_head 共享——语义上，"token → 向量空间"和"向量空间 → token 概率"用同一张映射表。

---

#### Fine-tune 部分：加载预训练权重

##### 理解的关键点

1. **state_dict 的本质：** PyTorch 模型就是一个 key→tensor 的字典。key 由属性名按层级拼接而来（`blocks.0.attn.c_attn.weight`）。加载权重 = 把别人字典里的值拷贝到你的字典里。
2. **HF 的 Conv1D vs nn.Linear：** 存储方式转置了。Linear 的 weight 是 (out, in)，Conv1D 是 (in, out)。所以 `c_attn.weight`, `c_proj.weight`, `c_fc.weight`, `c_proj.weight` 这四个加载时要 `.t()`。
3. **register_buffer 的角色：** 因果掩码不是参数，是常量。它出现在 state_dict 里但不需要从预训练模型拷贝（两边内容一样），所以要过滤掉。
4. **微调策略：** 学习率从 3e-4 降到 1e-5。预训练权重是好的起点，大学习率会破坏它（catastrophic forgetting）。
5. **Tokenizer 切换：** 从字符级（vocab ~65）切换到 GPT-2 的 BPE（vocab 50257）。模型的 embedding 层维度必须和 tokenizer 的词表大小一致。

---

#### 建立的心智模型

- **训练 = 不断调参数让 loss 下降。** loss 是 cross_entropy，衡量"模型预测的下一个 token 概率分布"和"真实下一个 token"的差距
- **模型本身只做一件事：** 输入 (B, T) 的 token 序列 → 输出 (B, T, vocab_size) 的logits(softmax归一化后即为概率)
- **生成 = 反复调用模型，每次取最后一个位置的概率，采样一个 token，拼回去**
- **所有的"理解"都编码在那些权重矩阵的数值里** ——加载预训练权重后模型立刻能用，就是因为那些数值已经编码了语言知识

