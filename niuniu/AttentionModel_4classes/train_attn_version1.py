import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import RNN
from CustomDataset import CustomDataset
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from attentionModel import Attention
torch.random.manual_seed(42)
np.random.seed(42)
random.seed(42)
train_losses = []

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}" )

    input_size = 3
    hidden_size = 64
    num_channels = 48
    num_classes = 4
    epochs = 80

    model = Attention(input_size=input_size, hidden_size=hidden_size, num_channels=num_channels, num_classes=num_classes)
    model = model.to(device)  # 将模型放到设备上
    
    # # 调优
    # # 加权损失函数：可以为 CrossEntropyLoss 设置类别权重，给“开启”和“关闭”类赋予更大的权重，从而让模型更加关注这些类别。
    # # 假设 labels 是训练集中所有标签
    # y_train_l = pd.read_csv("D:/niuniu/Dataset_version2/train_dataset/y_train.csv").values
    # labels = y_train_l.flatten()  # 将所有标签展平为一维数组
    # # 使用 compute_class_weight 计算权重
    # class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =0.00001)

    train_x_file = "D:/niuniu/Dataset_version2/train_dataset/x_train.csv"
    train_y_file = "D:/niuniu/Dataset_version2/train_dataset/y_train.csv"

    x_train = pd.read_csv(train_x_file).values
    y_train = pd.read_csv(train_y_file).values

    train_dataset = CustomDataset(data = x_train, labels=y_train, window_size=20)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle = False)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            output = output[:, -1, :, :]  # 最后一个时间步的输出
            output = output.reshape(-1, num_classes)  
            target = target.view(-1) 
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if torch.isnan(loss):
                print("Loss is NaN!")
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_model.pth")  # 保存最优模型权重
        
    torch.save(model.state_dict(), "final_model.pth")

    with open('train_losses.txt', 'w') as f:
        for loss in train_losses:
            f.write(f"{loss}\n")
    
    # 绘制 loss 曲线
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('D:/niuniu/Result_loss_fig/train_loss_curve.png')  
    plt.show()

if __name__ == "__main__":
    main()
