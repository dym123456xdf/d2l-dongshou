"""
=============================================================================
07线性回归的简洁实现（PyTorch版）
=============================================================================
本文件使用PyTorch的高级API（nn.Sequential、DataLoader、optim），
以更简洁的方式重新实现线性回归，替代从零开始的实现。
整体流程：生成数据 → 定义模型 → 训练 → 验证误差
=============================================================================
"""

# -------------------- 第一步：导入必要的库 --------------------
import numpy as np              # NumPy：数值计算库（虽主要用torch，d2l依赖numpy）
import torch                     # PyTorch：深度学习框架
from torch.utils import data     # data模块：提供Dataset和DataLoader数据迭代工具
from d2l_lib import torch as d2l  # d2l：本书封装的工具库（synthetic_data等）

# -------------------- 第二步：生成真实数据集 --------------------
# 设置真实的权重w和偏置b（模型试图学到的参数）
# torch.tensor([2, -3.4])：真实权重，两个特征对应的系数
true_w = torch.tensor([2, -3.4])
# 真实偏置，线性模型 y = X @ w + b 中的b
true_b = 4.2

# d2l.synthetic_data：根据真实参数生成合成的带噪声数据集
# 返回features（形状1000×2的tensor）和labels（形状1000×1的tensor）
# X = [x1, x2]，y = 2*x1 + (-3.4)*x2 + 4.2 + 噪声
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# -------------------- 第三步：定义数据迭代器 --------------------
def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器（DataLoader）。

    参数：
        data_arrays : tuple  — 包含(features, labels)的元组
        batch_size   : int    — 每个小批量的样本数量
        is_train     : bool   — 是否在每个epoch打乱数据（训练集为True）

    返回：
        DataLoader对象 — 可迭代，每次 yield 一个(batch_size,)的小批量
    """
    # data.TensorDataset：将特征和标签打包成PyTorch Dataset
    # *data_arrays 是解包，等价于 data.TensorDataset(features, labels)
    dataset = data.TensorDataset(*data_arrays)
    # data.DataLoader：数据加载器，支持批量、并行、shuffle等
    # shuffle=is_train：训练时打乱（避免模型按顺序学习），验证/测试时不打乱
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 设置小批量大小为10（每批处理10个样本）
batch_size = 10
# 构建数据迭代器：每次遍历data_iter可获取 (特征batch, 标签batch)
data_iter = load_array((features, labels), batch_size)

# next(iter(data_iter))：验证数据迭代器是否正常工作
# iter()将迭代器转为Python迭代器，next()取出第一个batch
# 若不报错说明数据加载器构造成功（返回两个tensor的元组）
next(iter(data_iter))

# -------------------- 第四步：定义模型 --------------------
# 从torch导入nn模块（neural network），提供各种网络层和损失函数
from torch import nn

# nn.Sequential：按顺序堆叠层的容器（类似乐高积木）
# nn.Linear(2, 1)：输入维度2，输出维度1的线性层（全连接层）
#   计算：y = X @ W^T + b，输入形状 (batch, 2)，输出形状 (batch, 1)
# net是一个包含单层的Sequential，net[0]即中间的nn.Linear层
net = nn.Sequential(nn.Linear(2, 1))

# -------------------- 第五步：初始化模型参数 --------------------
# net[0]是堆叠的第一层，即nn.Linear(2, 1)
# .weight.data：访问该层的权重矩阵（形状2×1）
# .normal_(0, 0.01)：用均值0、标准差0.01的正态分布覆盖初始化权重
net[0].weight.data.normal_(0, 0.01)
# .bias.data：访问该层的偏置向量（形状1，）
# .fill_(0)：用0填充偏置（偏置默认初始化为0）
net[0].bias.data.fill_(0)

# -------------------- 第六步：定义损失函数和优化算法 --------------------
# nn.MSELoss()：均方误差损失函数（Mean Squared Error）
#   loss = (1/n) * Σ(y_pred - y_true)²
#   用于回归任务，衡量预测值与真实值的平方误差
loss = nn.MSELoss()

# torch.optim.SGD：随机梯度下降优化器
# net.parameters()：返回模型所有可学习参数（权重和偏置），供优化器更新
# lr=0.03：学习率（learning rate），控制参数更新的步长
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# -------------------- 第七步：训练模型 --------------------
# 训练轮数：整个数据集被遍历的次数
num_epochs = 3

# 外层循环：遍历每个epoch
for epoch in range(num_epochs):
    # 内层循环：遍历data_iter中的每个小批量
    # 每次迭代X是一个(batch_size, 2)的特征batch，y是一个(batch_size,)的标签batch
    for X, y in data_iter:
        # 1. 前向传播：计算当前batch的预测值
        #   net(X)输出形状 (batch_size, 1)
        #   loss(net(X), y)计算预测值与真实值的均方误差，返回标量tensor
        l = loss(net(X), y)
        # 2. 梯度清零：PyTorch默认会累加梯度（便于RNN等场景），需手动清零
        trainer.zero_grad()
        # 3. 反向传播：根据损失l计算所有参数关于l的梯度
        #   调用后，所有可学习参数的.grad属性被填充
        l.backward()
        # 4. 参数更新：优化器根据梯度和学习率更新每个参数
        #   w_new = w_old - lr * gradient
        trainer.step()
    # 每个epoch结束后，在整个验证集/训练集上计算一次平均损失
    # 用于观察模型是否收敛（loss应逐 epoch下降）
    l = loss(net(features), labels)
    # f'...'：Python格式化字符串字面量（f-string）
    # {l:f}：将l转为浮点数字符串（不显示tensor结构，只显示数值）
    print(f'epoch {epoch + 1}, loss {l:f}')

# -------------------- 第八步：验证模型精度 --------------------
# 训练完成后，从模型中取出学到的参数
# net[0].weight.data：形状 (1, 2)，与true_w形状 (2,) 不同，需reshape
# .reshape(true_w.shape)：转为与true_w相同的形状 (2,) 方便对比
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))

# net[0].bias.data：形状 (1,)，与true_b形状 () 相同，直接使用
b = net[0].bias.data
print('b的估计误差：', true_b - b)
