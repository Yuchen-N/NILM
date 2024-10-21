import torch
from CustomDataset import CustomDataset
from model import RNN
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from attentionModel import Attention
batch_size = 8
input_size = 3
hidden_size = 64
num_channels = 48
num_classes = 4
epochs = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}" )
test_x_file= "D:/niuniu/Dataset/test_dataset/x_test.csv"
test_y_file = "D:/niuniu/Dataset/test_dataset/y_test.csv"

test_data = pd.read_csv(test_x_file).values
test_input = pd.read_csv(test_y_file).values

test_dataset =  CustomDataset(test_data, test_input, window_size = 30)
test_Dataloader = DataLoader(test_dataset, batch_size= batch_size,shuffle=False)
model = Attention(input_size=input_size, hidden_size=hidden_size, num_channels=num_channels, num_classes=num_classes)
pth_path = "D:/niuniu/80epochs_lr0.00001_attention_version1Dataset/best_model.pth"
model.load_state_dict(torch.load(pth_path))
model.to(device)  
model.eval()

all_targets = []
all_predictions = []

correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_Dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = output[:,-1,:,:]
        output = output.reshape(-1, num_classes)
        predictions = torch.argmax(output, dim=-1)
        predictions = predictions.cpu().numpy()
        target = target.view(-1).cpu().numpy()

        all_predictions.extend(predictions)
        all_targets.extend(target)



# 计算分类指标
conf_matrix = confusion_matrix(all_targets, all_predictions, labels=[0, 1,2,3])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("D:/niuniu/80epochs_lr0.00001_attention_version1Dataset/Evaluation/conf_heatmap.png")
plt.show()

accuracy = accuracy_score(all_targets, all_predictions)
precision = precision_score(all_targets, all_predictions, labels=[0, 1, 2,3], average='macro')
recall = recall_score(all_targets, all_predictions, labels=[0, 1,2,3], average='macro')
f1 = f1_score(all_targets, all_predictions, labels=[0, 1,2,3], average='macro')

# 打印结果
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# 假设 y_true 是真实标签, y_pred 是预测标签
print(classification_report(all_targets, all_predictions, target_names=['关闭状态','关闭','开启', '关闭状态'], labels=[0, 1,2,3]))

# 保存混淆矩阵
np.savetxt("D:/niuniu/80epochs_lr0.00001_attention_version1Dataset/Evaluation/confusion_matrix.csv", conf_matrix, delimiter=",", fmt="%d")