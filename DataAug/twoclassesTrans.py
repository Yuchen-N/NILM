import pandas as pd

# 读取标签数据
y_train_path = "D:/Ai/NILM/Dataset/train_dataset/y_train.csv"
y_test_path = "D:/Ai/NILM/Dataset/test_dataset/y_test.csv"
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)

# 假设 2 表示开启状态，0 或者 3 表示关闭状态，-1 也作为关闭状态
# 2 映射为 1（开启），0 和 3 映射为 0（关闭）
y_train.replace({3: 0, 0: 0, 1: 1, 2: 1}, inplace=True)
y_test.replace({3: 0, 0: 0, 1: 1, 2: 1}, inplace=True)

y_testpath = "D:/Ai/NILM/Dataset_version2/test_dataset/y_test.csv"
y_trainpath = "D:/Ai/NILM/Dataset_version2/train_dataset/y_train.csv"

y_test.to_csv(y_testpath, index=False)  
y_train.to_csv(y_trainpath, index=False)

print("标签已成功转换并保存。")
