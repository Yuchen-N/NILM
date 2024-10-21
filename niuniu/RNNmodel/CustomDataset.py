import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, window_size = 1 ):
        self.data = data
        self.labels = labels
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.window_size]
        y = self.labels[index + self.window_size -1]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
