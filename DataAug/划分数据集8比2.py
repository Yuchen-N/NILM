import pandas as pd
import os

x_file_path= "D:/Ai/NILM/DALE_pressDataset/all_channels_combined.csv"
y_file_path = "D:/Ai/NILM/ProcessedDataset/main_resampled.csv"
testDataset_path = "D:/Ai/NILM/Dataset/test_dataset/"
trainDataset_path ="D:/Ai/NILM/Dataset/train_dataset/"
train_size = 0.8  # 80% 数据用于训练，20% 用于测试

x_df = pd.read_csv(x_file_path)
y_df = pd.read_csv(y_file_path)

x_df = x_df.set_index('timestamp')
y_df = y_df.set_index('timestamp')


split_idx = int(len(x_df) * train_size)

x_train = x_df[190393:391260]
y_train = y_df[190393:391260]


x_test = x_df[664121:671121]
y_test = y_df[664121:671121]

train_x = os.path.join(trainDataset_path, "y_train.csv")
train_y = os.path.join(trainDataset_path, "x_train.csv")

test_x = os.path.join(testDataset_path,"y_test.csv")
test_y = os.path.join(testDataset_path, "x_test.csv")
print(f"X 的训练集大小: {x_train.shape}, 测试集大小: {x_test.shape}")
print(f"Y 的训练集大小: {y_train.shape}, 测试集大小: {y_test.shape}")

x_train.to_csv(train_x, index=False)
y_train.to_csv(train_y, index=False)

x_test.to_csv(test_x, index=False)
y_test.to_csv(test_y, index = False)
