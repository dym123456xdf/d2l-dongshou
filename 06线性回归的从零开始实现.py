"""
线性回归的从零开始实现
======================
本文件演示如何使用PyTorch从零实现线性回归模型，包括：
- 生成合成数据集
- 定义模型、损失函数和优化算法
- 训练过程可视化
"""

# ============================================================
# 1. 导入必要的库
# ============================================================
import random          # 用于随机打乱数据索引
import torch           # PyTorch深度学习框架
from d2l_lib import torch as d2l  # 动手学深度学习教材配套库（绘图、数据处理等）


# ============================================================
# 2. 生成合成数据集
# ============================================================
def synthetic_data(w, b, num_examples):
    """
    生成合成的线性回归数据集。

    参数:
        w (torch.Tensor): 真实的权重向量，形状为 (num_features, 1)
        b (float): 真实的偏置项
        num_examples (int): 生成的样本数量

    返回:
        X (torch.Tensor): 特征矩阵，形状为 (num_examples, num_features)
        y (torch.Tensor): 标签向量，形状为 (num_examples, 1)

    数学原理:
        生成的公式为: y = Xw + b + noise
        其中noise服从均值为0、标准差为0.01的正态分布
    """
    # 生成均值为0、标准差为1的正态分布随机数作为特征矩阵
    # 形状: (num_examples, len(w))，len(w)是特征维度
    X = torch.normal(0, 1, (num_examples, len(w)))

    # 计算真实的标签值: y = X @ w + b
    # torch.matmul(X, w) 执行矩阵乘法
    y = torch.matmul(X, w) + b

    # 添加噪声项，模拟真实世界数据的随机误差
    # 噪声服从均值为0、标准差为0.01的正态分布
    y += torch.normal(0, 0.01, y.shape)

    # 将y转换为列向量形式 (num_examples, 1)
    return X, y.reshape((-1, 1))


# 设置随机种子，确保结果可复现
# 这样每次运行都会生成相同的随机数序列
torch.manual_seed(42)

# 定义真实的模型参数（我们希望通过训练学到的）
true_w = torch.tensor([2, -3.4])  # 真实权重: [2, -3.4]
true_b = 4.2                       # 真实偏置: 4.2

# 生成1000个训练样本
num_examples = 1000
features, labels = synthetic_data(true_w, true_b, num_examples)

# 打印第一个样本的特征和标签，验证数据生成正确
print("=" * 60)
print("数据集生成验证")
print("=" * 60)
print(f"特征矩阵形状: {features.shape}")  # 应该是 (1000, 2)
print(f"标签向量形状: {labels.shape}")    # 应该是 (1000, 1)
print(f"第一个样本特征: {features[0]}")
print(f"第一个样本标签: {labels[0]}")


# ============================================================
# 3. 可视化数据集
# ============================================================
# 使用d2l库设置图形大小并绘制散点图
# features[:, 1] 取所有样本的第2列特征（索引从0开始）
d2l.set_figsize()  # 设置matplotlib图形大小
d2l.plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(),
                1)  # 绘制散点图，点大小为1
d2l.plt.show()  # 显示图形


# ============================================================
# 4. 定义数据迭代器
# ============================================================
def data_iter(batch_size, features, labels):
    """
    小批量随机梯度下降的数据迭代器。

    参数:
        batch_size (int): 每个小批量的样本数量
        features (torch.Tensor): 特征矩阵
        labels (torch.Tensor): 标签向量

    生成:
        (X_batch, y_batch): 小批量的特征和标签

    工作原理:
        1. 计算样本总数
        2. 创建随机排列的索引
        3. 每次返回batch_size个样本
    """
    # 计算样本总数
    num_examples = len(features)

    # 创建索引列表: [0, 1, 2, ..., num_examples-1]
    indices = list(range(num_examples))

    # 随机打乱索引，实现数据的随机采样
    # 这确保每个epoch中样本的顺序是随机的
    random.shuffle(indices)

    # 每次取batch_size个样本
    # i 是起始索引，每次步长为batch_size
    for i in range(0, num_examples, batch_size):
        # 计算当前批次的结束索引
        # 最后一批可能不足batch_size个样本
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])

        # 使用高级索引获取对应的特征和标签
        yield features[batch_indices], labels[batch_indices]


# 设置批量大小
batch_size = 10

# 测试数据迭代器，打印第一批数据
print("\n" + "=" * 60)
print("数据迭代器测试（第一批数据）")
print("=" * 60)
for X, y in data_iter(batch_size, features, labels):
    print(f"批次特征形状: {X.shape}")  # 应该是 (10, 2)
    print(f"批次标签形状: {y.shape}")   # 应该是 (10, 1)
    print(f"特征 X:\n{X}")
    print(f"标签 y:\n{y}")
    break  # 只打印第一批就退出循环


# ============================================================
# 5. 初始化模型参数
# ============================================================
print("\n" + "=" * 60)
print("模型参数初始化")
print("=" * 60)

# 初始化权重 w：将权重初始化为均值为0、标准差为0.01的正态分布随机数
# 形状为 (2, 1)，与真实权重的形状相同
# requires_grad=True 表示需要计算梯度，用于反向传播
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)

# 初始化偏置 b：将偏置初始化为0
# 形状为 (1,)，标量
b = torch.zeros(1, requires_grad=True)

print(f"初始权重 w:\n{w}")
print(f"初始偏置 b: {b}")


# ============================================================
# 6. 定义线性回归模型
# ============================================================
def linreg(X, w, b):
    """
    线性回归模型的前向传播计算。

    参数:
        X (torch.Tensor): 输入特征，形状为 (batch_size, num_features)
        w (torch.Tensor): 权重，形状为 (num_features, 1)
        b (torch.Tensor): 偏置，形状为 (1,) 或 (1, 1)

    返回:
        torch.Tensor: 预测值，形状为 (batch_size, 1)

    数学公式:
        y_pred = X @ w + b
        其中 @ 表示矩阵乘法
    """
    # 矩阵乘法: (batch_size, num_features) @ (num_features, 1)
    #           -> (batch_size, 1)
    return torch.matmul(X, w) + b


# ============================================================
# 7. 定义损失函数（均方损失）
# ============================================================
def squared_loss(y_hat, y):
    """
    计算均方损失函数（MSE Loss）。

    参数:
        y_hat (torch.Tensor): 模型预测值，形状为 (batch_size, 1)
        y (torch.Tensor): 真实标签，形状为 (batch_size, 1) 或 (batch_size,)

    返回:
        torch.Tensor: 每个样本的损失值，形状与y_hat相同

    数学公式:
        L = (1/2) * (y_hat - y)^2

    注意:
        除以2是为了在求导时消去平方项的系数，简化计算
    """
    # 将y reshape成与y_hat相同的形状，然后计算均方损失
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# ============================================================
# 8. 定义优化算法（小批量随机梯度下降）
# ============================================================
def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降（SGD）优化器。

    参数:
        params (list): 模型参数列表 [w, b]
        lr (float): 学习率（learning rate）
        batch_size (int): 批量大小，用于归一化梯度

    更新公式:
        param = param - lr * gradient / batch_size

    工作原理:
        1. 使用torch.no_grad()临时禁用梯度计算，节省内存和计算
        2. 对每个参数执行原位更新（in-place update）
        3. 更新完成后，手动将梯度清零（为下一次反向传播做准备）
    """
    # 禁用梯度计算，因为这是参数更新阶段，不需要梯度追踪
    with torch.no_grad():
        for param in params:
            # 梯度下降更新: param_new = param_old - lr * grad
            # 注意：这里除以batch_size是因为损失是所有样本损失的平均
            param -= lr * param.grad / batch_size

            # 重要：每次参数更新后必须将梯度清零
            # 否则梯度会累积，导致下次反向传播时梯度计算错误
            param.grad.zero_()


# ============================================================
# 9. 训练模型
# ============================================================
print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

# 设置超参数
lr = 0.03       # 学习率：控制参数更新的步长
num_epochs = 3  # 训练轮数：整个数据集被遍历的次数

# 训练循环
for epoch in range(num_epochs):
    # 每个epoch遍历一次数据集
    for X, y in data_iter(batch_size, features, labels):
        # ----- 前向传播 -----
        # 计算当前批次的预测值
        y_hat = linreg(X, w, b)

        # 计算当前批次的损失
        l = squared_loss(y_hat, y)

        # ----- 反向传播 -----
        # 计算梯度
        # 注意：backward()会累积梯度到.grad属性中
        l.sum().backward()  # 对损失求和后反向传播

        # ----- 参数更新 -----
        # 使用SGD更新参数
        sgd([w, b], lr, batch_size)

    # ----- 验证（每个epoch结束后） -----
    # 使用整个数据集计算训练损失，评估模型性能
    with torch.no_grad():
        # 计算整个数据集上的预测
        y_hat_all = linreg(features, w, b)

        # 计算整个数据集上的平均损失
        train_l = squared_loss(y_hat_all, labels)

        # 打印训练进度
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"损失 (MSE): {float(train_l.mean()):.6f}")

# ============================================================
# 10. 输出训练结果
# ============================================================
print("\n" + "=" * 60)
print("训练结果")
print("=" * 60)

# 比较学到的参数与真实参数的差异
w_error = true_w - w.reshape(true_w.shape)
b_error = true_b - b

print(f"真实权重 w: {true_w}")
print(f"学习到的权重: {w.reshape(true_w.shape)}")
print(f"权重估计误差: {w_error}")

print(f"\n真实偏置 b: {true_b}")
print(f"学习到的偏置: {b}")
print(f"偏置估计误差: {b_error}")

# 打印最终损失
with torch.no_grad():
    final_loss = squared_loss(linreg(features, w, b), labels)
    print(f"\n最终训练损失 (MSE): {float(final_loss.mean()):.6f}")

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
