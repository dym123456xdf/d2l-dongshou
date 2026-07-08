import torch

# =============================================================================
# 第1节：张量基础 - torch.tensor() 创建张量
# =============================================================================
# torch.tensor([3.0]) 中的 [] 表示创建 1维张量（形状为 (1,)），包含1个元素的向量
# 如果不加 []，如 torch.tensor(3.0)，则是标量（0维）
# 深度学习中通常使用张量而非标量，因为需要处理批量数据
# 逐元素运算（element-wise operations）
print("=" * 60)
print("第1节：张量基础 - torch.tensor() 创建张量")
print("=" * 60)
x = torch.tensor([3.0])
y = torch.tensor([2.0])
# 加法: 3.0 + 2.0 = 5.0
print(f"x + y = {x + y}")
# 乘法: 3.0 * 2.0 = 6.0
print(f"x * y = {x * y}")
# 除法: 3.0 / 2.0 = 1.5
print(f"x / y = {x / y}")
# 幂运算: 3.0 ** 2.0 = 9.0
print(f"x**y = {x ** y}")

# =============================================================================
# 第2节：torch.arange() 创建序列张量
# =============================================================================
# torch.arange(4) 生成从0开始、步长为1的张量，结果为 [0, 1, 2, 3]
# 类似于 Python 的 range(4)，但返回张量
print("\n" + "=" * 60)
print("第2节：torch.arange() 创建序列张量")
print("=" * 60)
x = torch.arange(4)
print(f"x = {x}")
# 访问张量中的单个元素，索引从0开始（第4个元素，值为3）
print(f"x[3] = {x[3]}")
# len() 返回张量的元素个数
print(f"len(x) = {len(x)}")
# shape 返回张量的形状（各维度的大小）
print(f"x.shape = {x.shape}")

# =============================================================================
# 第3节：二维张量与转置
# =============================================================================
# torch.arange(20) 生成0到19的数字
# reshape(5, 4) 将其调整为5行4列的二维张量
# shape (5, 4) 表示：5行，4列
print("\n" + "=" * 60)
print("第3节：二维张量与转置")
print("=" * 60)
A = torch.arange(20).reshape(5, 4)
print(f"A = torch.arange(20).reshape(5, 4) = \n{A}")
# .T 转置：将矩阵的行和列互换
# 原形状 (5, 4) -> 转置后形状 (4, 5)
print(f"A.T (转置) = \n{A.T}")

# =============================================================================
# 第4节：对称矩阵
# =============================================================================
# 对称矩阵：满足 B[i,j] == B[j,i] 的方阵
# B == B.T 是逐元素比较，返回布尔张量
print("\n" + "=" * 60)
print("第4节：对称矩阵")
print("=" * 60)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(f"B = \n{B}")
print(f"B == B.T = \n{B == B.T}")
# 结果全为 True，说明 B 是对称矩阵

# =============================================================================
# 第5节：三维张量
# =============================================================================
# reshape(2, 3, 4) 创建3维张量
# shape (2, 3, 4) 表示：2个3×4的矩阵（可以理解为2个样本，每个样本3行4列）
# 常用于批量数据，如 batch_size=2, height=3, width=4

print("\n" + "=" * 60)
print("第5节：三维张量")
print("=" * 60)
X = torch.arange(24).reshape(2, 3, 4)
print(f"X = torch.arange(24).reshape(2, 3, 4) = \n{X}")
print(f"X.shape = {X.shape}")

# =============================================================================
# 第6节：张量克隆（深拷贝）
# =============================================================================
# clone() 创建张量的深拷贝，修改新张量不会影响原张量（不共享内存）
# dtype=torch.float32 指定数据类型为32位浮点数
print("\n" + "=" * 60)
print("第6节：张量克隆（深拷贝）")
print("=" * 60)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(f"A = \n{A}")
print(f"B = A.clone() = \n{B}")

# 逐元素相加
print(f"A + B = \n{A + B}")

# 逐元素乘法（非矩阵乘法，对应位置相乘）
print(f"A * B (逐元素乘法) = \n{A * B}")

# =============================================================================
# 第7节：标量与张量的广播运算
# =============================================================================
# 广播（broadcasting）：标量可以自动扩展到与张量相同的形状进行运算
# 标量 a=2 会广播到 X 的每个元素

print("\n" + "=" * 60)
print("第7节：标量与张量的广播运算")
print("=" * 60)
a = 2
X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(f"X = \n{X}")
print(f"a = {a}, X.shape = {X.shape}")
print(f"a + X (标量广播) = \n{a + X}")
print(f"(a * X).shape = {(a * X).shape}")  # 形状不变，仍为 (2, 3, 4)

# =============================================================================
# 第8节：张量求和
# =============================================================================
# .sum() 对张量所有元素求和，返回标量
print("\n" + "=" * 60)
print("第8节：张量求和")
print("=" * 60)
x = torch.arange(4, dtype=torch.float32)
print(f"x = {x}")
print(f"x.sum() = {x.sum()}")  # 0 + 1 + 2 + 3 = 6

A = torch.arange(20).reshape(5, 4)
print(f"A = \n{A}")
print(f"A.shape = {A.shape}")
print(f"A.sum() = {A.sum()}")  # 0到19相加 = 190

# =============================================================================
# 第9节：按轴（axis）求和
# =============================================================================
# axis=0：沿第0维（列）求和，结果形状 (5, 4) -> (4,)
#         即每列的元素相加，得到每列的总和
print("\n" + "=" * 60)
print("第9节：按轴（axis）求和")
print("=" * 60)
A = torch.arange(20).reshape(5, 4)
print(f"A = \n{A}")
A_sum_axis0 = A.sum(axis=0)
print(f"A = \n{A}")
print(f"A.sum(axis=0) = {A_sum_axis0}")  # 每列求和
print(f"A_sum_axis0.shape = {A_sum_axis0.shape}")

# axis=1：沿第1维（行）求和，结果形状 (5, 4) -> (5,)
#         即每行的元素相加，得到每行的总和

A_sum_axis1 = A.sum(axis=1)
print(f"A.sum(axis=1) = {A_sum_axis1}")  # 每行求和
print(f"A_sum_axis1.shape = {A_sum_axis1.shape}")

# axis=[0, 1]：同时沿两个轴求和，结果为标量，等价于 A.sum()
print(f"A.sum(axis=[0, 1]) = {A.sum(axis=[0, 1])}")

# =============================================================================
# 第10节：均值与非均值求和
# =============================================================================
# .mean() 计算张量所有元素的平均值
# .numel() 返回张量的元素总数
# 验证：mean() 等于 sum() / numel()
print("\n" + "=" * 60)
print("第10节：均值与非均值求和")
print("=" * 60)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(f"A = \n{A}")
print(f"A.mean() = {A.mean()}")  # 所有元素的平均值
print(f"A.sum() / A.numel() = {A.sum() / A.numel()}")  # 验证均值

# 按axis=0（列）求均值，验证：mean(axis=0) 等于 sum(axis=0) / shape[0]
print(f"A.mean(axis=0) = {A.mean(axis=0)}")  # 每列均值
print(f"A.sum(axis=0) / A.shape[0] = {A.sum(axis=0) / A.shape[0]}")  # 验证

# =============================================================================
# 第11节：累加求和（cumsum）
# =============================================================================
# .cumsum(axis=0) 沿指定轴的累加和，每元素是到该位置为止的累计和
# keepdims=True 保持维度，使得广播运算时形状兼容

print("\n" + "=" * 60)
print("第11节：累加求和（cumsum）与 keepdims")
print("=" * 60)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(f"A = \n{A}")

# 按行累加，keepdims=True 保持二维形状 (5, 1)，便于广播除法
sum_A = A.sum(axis=1, keepdims=True)  # shape: (5, 4) -> (5, 1)
print(f"A.sum(axis=1, keepdims=True) = \n{sum_A}")
print(f"sum_A.shape = {sum_A.shape}")

# 广播：A 的每行除以该行的总和（归一化）
print(f"A / sum_A (每行归一化) = \n{A / sum_A}")

# 按axis=0（列）累加，不保持维度
print(f"A = \n{A}")
print(f"A.cumsum(axis=0) = \n{A.cumsum(axis=0)}")

# =============================================================================
# 第12节：点积（Dot Product）
# =============================================================================
# torch.dot(x, y) 计算两个向量的点积：对应元素相乘后求和
# 等价于 torch.sum(x * y)
# 点积结果是一个标量

print("\n" + "=" * 60)
print("第12节：点积（Dot Product）")
print("=" * 60)
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(f"x = {x}")
print(f"y = {y}")
print(f"torch.dot(x, y) = {torch.dot(x, y)}")  # 1*4 + 2*5 + 3*6 = 32
print(f"torch.sum(x * y) = {torch.sum(x * y)}")  # 验证：结果相同

# =============================================================================
# 第13节：矩阵-向量积（Matrix-Vector Product）
# =============================================================================
# torch.mv(A, x) 计算矩阵A与向量x的乘积
# 矩阵A的列数必须等于向量x的元素个数
# 结果是一个向量，形状为 A的行数

print("\n" + "=" * 60)
print("第13节：矩阵-向量积（Matrix-Vector Product）")
print("=" * 60)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print(f"A = \n{A}")
print(f"x = {x}")
print(f"A.shape = {A.shape}")  # (5, 4)
print(f"x.shape = {x.shape}")  # (4,)
print(f"torch.mv(A, x) = {torch.mv(A, x)}")
# 计算过程：每行与x点积
# 第0行: 0*1 + 1*2 + 2*3 + 3*4 = 0+2+6+12 = 20

# =============================================================================
# 第14节：矩阵-矩阵乘法（Matrix-Matrix Multiplication）
# =============================================================================
# torch.mm(A, B) 计算矩阵A与矩阵B的乘积
# A的列数必须等于B的行数
# 结果形状：(A的行数, B的列数)

print("\n" + "=" * 60)
print("第14节：矩阵-矩阵乘法（Matrix-Matrix Multiplication）")
print("=" * 60)
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)   # 2x3矩阵
B = torch.arange(12, dtype=torch.float32).reshape(3, 4)  # 3x4矩阵
print(f"A = \n{A}")
print(f"A.shape = {A.shape}")      # (2, 3)
print(f"B = \n{B}")
print(f"B.shape = {B.shape}")      # (3, 4)
print(f"torch.mm(A, B) = \n{torch.mm(A, B)}")  # 2x4矩阵
print(f"torch.mm(A, B).shape = {torch.mm(A, B).shape}")
