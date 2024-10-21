import pandas as pd

# 读取 y_train.csv 文件
y_test_path = "D:/Ai/NILM/Dataset/train_dataset/y_train.csv"
y_test = pd.read_csv(y_test_path)
# 映射标签
# -1 -> 3, 0 -> 0, 1 -> 1, 2 -> 2
y_test.replace({-1: 3, 0: 0, 1: 1, 2: 2}, inplace=True)

# 保存映射后的 y_train.csv 文件
y_test.to_csv(y_test_path, index=False)

print("标签映射完成并保存至 y_train.csv 文件")
