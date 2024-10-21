import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels, num_classes, num_heads =4):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,batch_first=True)
        self.fc = nn.Linear(hidden_size, num_channels*num_classes)
        self.fc_in = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.fc_in(x)
        attn_output, _ = self.attention(x,x,x)
        
        out_fc = self.fc(attn_output)
        out_reshaped = out_fc.view(-1, attn_output.size(1), self.num_channels, self.num_classes)
        return out_reshaped
