# **动手实现built-nanogpt项目**

## **实现路径**
当前对transformer建立了模糊理解，开始着手实现nanogpt以厘清细节，并初步掌握Pytorch库。
完成项目的路径为：
- 已有: 线性代数 + 微积分
- 需要建立: 统计学习基础（loss函数、概率模型的直觉）
- 需要建立: 神经网络基础（前向传播、反向传播、梯度下降）
- 需要建立: 深度学习组件（Embedding、归一化、激活函数）
- 目标: Transformer 架构（Attention机制、GPT结构）
- 最终: 读懂并推演 nanogpt 代码

## **nanogpt项目的提交记录分层**

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

## **3.28**

- 尝试跟视频阅读源码，Pytorch代码一窍不通
- 学习Pytorch但不熟悉向前传播、feature等基本概念
- 明确**阶段学习路径**：
1. 读懂Python中的`__call__`与运算符重载，掌握神经网络基础概念
2. 学习[micrograd](https://github.com/karpathy/micrograd)项目 + 跟着手写代码
3. 进入 build-nanogpt

### `__call__`与运算符重载

- `__call__`:一种“魔术方法”，使得**对象可以像函数一样被调用**，如定义了`__call__`的对象`obj`，`obj(x)`等价于`obj.__call__(x)`
- 运算符重载：通过定义“魔术方法”，可以**自定义对象的内置运算符**，如+-*/等的行为
  
## **3.29**

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
