import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels, num_classes, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,num_channels*num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out_fc = self.fc(out)  # 形状: (batch_size, seq_len, num_channels * num_classes)
        out_reshaped = out_fc.view(-1, out.size(1), self.num_channels, self.num_classes)

        return out_reshaped