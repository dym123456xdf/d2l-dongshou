"""
线性回归示例脚本
本文件包含两个演示：
1. 标量逐元素运算 vs 向量批量运算的性能对比（计时器）
2. 正态分布（高斯分布）概率密度函数的绘制
"""

# ============================================================
# 第一部分：导入标准库和第三方库
# ============================================================

# 导入数学库，提供 sqrt、pi、exp 等数学函数
import math

# 导入时间库，用于记录程序运行时间
import time

# 导入 NumPy 库，简写为 np，用于数值计算
import numpy as np

# 导入 PyTorch 深度学习框架
import torch

# 导入 Matplotlib 的 pyplot 模块，简写为 plt，用于绘图
import matplotlib.pyplot as plt

# ============================================================
# 第二部分：创建 PyTorch 向量（张量）
# ============================================================

# 定义向量长度 n = 10000，即每个向量包含 10000 个元素
n = 10000

# 创建全 1 张量 a，长度为 n，用于后续加法运算演示
a = torch.ones(n)

# 创建全 1 张量 b，长度为 n，与 a 形状相同
b = torch.ones(n)

# ============================================================
# 第三部分：定义计时器类 Timer
# ============================================================

# 定义计时器类：用于记录多次操作的运行时间，典型使用场景是比较两种算法的性能开销
class Timer:
    # 创建计时器对象时调用，初始化存储列表并自动启动计时
    def __init__(self):
        # 初始化空列表，用于存储每次计时的结果（秒）
        self.times = []
        # 创建对象时自动启动计时器
        self.start()

    # 启动计时器：记录当前时间戳到 self.tik
    def start(self):
        # time.time() 返回自 1970-01-01 以来的秒数（Unix 时间戳）
        self.tik = time.time()

    # 停止计时器：将从 start() 到此刻的时间差记录到 self.times 列表中
    def stop(self):
        # 当前时间减去启动时间，即为经过的时长，追加到列表中
        self.times.append(time.time() - self.tik)
        # 返回列表中最后一个元素，即本次计时结果
        return self.times[-1]

    # 返回所有已记录时长的平均值（秒）
    def avg(self):
        return sum(self.times) / len(self.times)

    # 返回所有已记录时长的总和（秒）
    def sum(self):
        return sum(self.times)

    # 返回累计时间列表，cumsum 是 cumulative sum（累计和）的缩写
    def cumsum(self):
        # 将列表转换为 NumPy 数组，调用 cumsum() 计算累计和，再转回 Python 列表
        return np.array(self.times).cumsum().tolist()

# ============================================================
# 第四部分：性能对比 — 逐元素循环 vs 向量化运算
# ============================================================

# 创建全 0 张量 c，长度为 n，用于存储逐元素相加的结果
c = torch.zeros(n)

# 创建 Timer 类的实例 timer，开始自动计时
timer = Timer()

# for 循环，遍历索引 0 到 n-1（共 n 次迭代）
for i in range(n):
    # 将向量 a 和 b 的第 i 个元素相加，结果存入 c 的第 i 个位置
    # 此方式为 Python 逐元素操作，效率较低（需循环 n 次）
    c[i] = a[i] + b[i]

# 停止计时，返回字符串格式的时长，保留 5 位小数（单位：秒）
# f-string 格式：.5f 表示格式化浮点数，保留小数点后 5 位
f'{timer.stop():.5f} sec'

# 重新启动计时器，为第二次测试计时
timer.start()

# PyTorch 张量重载了 + 运算符，自动进行逐元素相加（向量化运算）
# 向量化运算由底层 C/CUDA 实现，无需 Python 循环，效率远高于逐元素循环
d = a + b

# 停止计时，返回向量化运算的耗时
f'{timer.stop():.5f} sec'

# ============================================================
# 第五部分：正态分布（高斯分布）概率密度函数
# ============================================================

# 定义正态分布（高斯分布）的概率密度函数（PDF）
def normal(x, mu, sigma):
    """
    参数：
        x     : 输入值，可以是标量或数组（NumPy ndarray）
        mu    : 正态分布的均值（mean），决定分布曲线的中心位置
        sigma : 正态分布的标准差（standard deviation），决定分布曲线的宽度

    返回：
        p : 输入 x 对应的正态分布概率密度值

    数学公式：
        p(x) = (1 / sqrt(2 * pi * sigma^2)) * exp(-0.5 * (x - mu)^2 / sigma^2)

    物理意义：
        均值 mu 决定分布中心，标准差 sigma 决定分布的宽窄
        sigma 越大，曲线越平坦；sigma 越小，曲线越尖锐
    """
    # 第一项：系数因子 (1 / sqrt(2 * pi * sigma^2))，确保整个分布曲线下面积为 1
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    # 第二项：指数因子 exp(-0.5 * (x - mu)^2 / sigma^2)，决定曲线的钟形形状
    p = p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
    # 返回概率密度值
    return p

# 生成 x 轴数据：从 -7 到 7（不包含 7），步长 0.01
# np.arange(start, stop, step) 生成等差数列，共约 1400 个点，保证曲线光滑
x = np.arange(-7, 7, 0.01)

# 定义多组正态分布参数 (均值 mu, 标准差 sigma)，共三组：
#   (0, 1) ：标准正态分布，均值为 0，标准差为 1
#   (0, 2) ：均值 0，标准差 2，曲线更宽更平坦
#   (3, 1) ：均值 3，标准差 1，曲线形状与标准正态相同但中心在 x=3
params = [(0, 1), (0, 2), (3, 1)]

# 依次绘制三条正态分布曲线
for mu, sigma in params:
    # plt.plot：在二维坐标系中绘制曲线
    # x                     ：x 轴数据
    # normal(x, mu, sigma) ：y 轴数据（概率密度）
    # label                 ：图例标签，显示该曲线对应的均值和标准差
    plt.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}')

# 设置 x 轴的标签文字
plt.xlabel('x')

# 设置 y 轴的标签文字（概率密度）
plt.ylabel('p(x)')

# 显示图例，将每条曲线的 label 信息展示出来
plt.legend()

# 调用 Matplotlib 显示图像窗口，程序会在此处暂停等待用户关闭图像
plt.show()
