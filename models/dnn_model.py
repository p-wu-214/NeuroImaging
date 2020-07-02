import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        x = self.linear(x)

        return x

class TrendsDNNModel(nn.Module):
    def __init__(self):
        super(TrendsDNNModel, self).__init__()
        self.linear1 = BasicBlock(1404, 5)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return x