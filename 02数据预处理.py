# 导入os模块，用于处理文件路径和目录操作
import os

# 在当前目录下创建data文件夹；exist_ok=True表示若已存在则不报错
os.makedirs(os.path.join('.', 'data'), exist_ok=True)

# 拼接CSV文件的完整路径：./data/house_tiny.csv
data_file = os.path.join('.', 'data', 'house_tiny.csv')

# 以写入模式('w')打开文件，使用with语句自动关闭文件
with open(data_file, 'w') as f:
    # 写入CSV表头：NumRooms(房间数)、Alley(巷道类型)、Price(房价)
    f.write('NumRooms,Alley,Price\n')
    # 第1行数据：房间数缺失(NA)，巷道类型为Pave(有巷道)，房价127500
    f.write('NA,Pave,127500\n')
    # 第2行数据：房间数2，巷道类型缺失(NA表示无巷道)，房价106000
    f.write('2,NA,106000\n')
    # 第3行数据：房间数4，巷道类型缺失(NA表示无巷道)，房价178100
    f.write('4,NA,178100\n')
    # 第4行数据：房间数缺失(NA)，巷道类型缺失(NA表示无巷道)，房价140000
    f.write('NA,NA,140000\n')

# 读取CSV数据
import pandas as pd
data = pd.read_csv(data_file)
print("原始数据:")
print(data)

# 分离数值列和类别列
# inputs = data.iloc[:, 0:2]  # 前两列
# outputs = data.iloc[:, 2]   # 第3列（房价）

# 提取数值列和类别列
# 数值列：NumRooms（房间数），用均值填充缺失值
num_cols = data.iloc[:, 0:1]  # 第1列NumRooms
num_cols = num_cols.fillna(num_cols.mean())

# 类别列：Alley（巷道类型），转换为独热编码
cat_cols = data.iloc[:, 1:2]  # 第2列Alley
print("原始Alley列：")
print(cat_cols)
print()

# 对比1：dummy_na=True（包含NA缺失值列）
cat_dummies_with_na = pd.get_dummies(cat_cols, dummy_na=True)
print("dummy_na=True（包含NA列）：")
print(cat_dummies_with_na)
print()

# 对比2：dummy_na=False（不包含NA缺失值列）
cat_dummies_without_na = pd.get_dummies(cat_cols, dummy_na=False)
print("dummy_na=False（不包含NA列）：")
print(cat_dummies_without_na)

# 合并处理后的数值列和类别列
# dummy_na=True的独热编码结果（包含NA列），更完整地保留缺失信息
inputs = pd.concat([num_cols, cat_dummies_with_na], axis=1)
outputs = data.iloc[:, 2]

print("\n最终处理后的inputs（dummy_na=True）：")
print(inputs)
print("\noutputs:")
print(outputs)


import torch

# 将布尔值转换为数值（True->1, False->0），再转为tensor
X, y = torch.tensor(inputs.values.astype(float)), torch.tensor(outputs.values.astype(float))
print("X (特征):")
print(X)
print("\ny (标签):")
print(y)
