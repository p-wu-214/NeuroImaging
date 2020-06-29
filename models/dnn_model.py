import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)

        return x


class SBMDnnModel(nn.Module):
    def __init__(self):
        super(SBMDnnModel, self).__init__()
        self.layer1 = BasicBlock(26, 26)
        self.layer2 = BasicBlock(26, 26)
        self.batch = nn.BatchNorm1d(26)
        # self.layer3 = BasicBlock(15, 5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batch(x)

        return x

class FNCDnnModel(nn.Module):
    def __init__(self):
        super(FNCDnnModel, self).__init__()
        self.layer1 = BasicBlock(1378, 1378)
        self.layer2 = BasicBlock(1378, 1378)
        self.batch = nn.BatchNorm1d(1378)
        # self.layer3 = BasicBlock(1378, 1378)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batch(x)

        return x

class TrendsDNNModel(nn.Module):
    def __init__(self):
        super(TrendsDNNModel, self).__init__()
        self.fnc_layer = FNCDnnModel()
        self.sbm_layer = SBMDnnModel()
        self.linear1 = BasicBlock(1404, 705)
        self.linear2 = BasicBlock(705, 705)
        self.linear3 = BasicBlock(705, 5)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fnc_data, sbm_data):
        x1 = self.fnc_layer(fnc_data)
        x2 = self.sbm_layer(sbm_data)
        x = torch.cat((x1, x2), dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.relu(x)
        return x