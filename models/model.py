from models.dnn_model import TrendsDNNModel
from models.resnet_model import ResNetBasicBlock

import torch
import torch.nn as nn

class NeuroImageModel(nn.Module):

    def __init__(self):
        super(NeuroImageModel, self).__init__()
        self.dnn_output = TrendsDNNModel()
        self.mri_output = ResNetBasicBlock()
        self.linear1 = nn.Linear(in_features=10, out_features=10, bias=True)
        self.linear2 = nn.Linear(in_features=10, out_features=5, bias=True)

    def forward(self, fnc_data, sbm_data, x):
        x1 = self.dnn_output(fnc_data, sbm_data)
        x2 = self.mri_output(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
