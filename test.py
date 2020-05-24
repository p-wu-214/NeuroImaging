import gcsfs
import torch
from torch import nn
if __name__ == '__main__':
    m = nn.AdaptiveAvgPool3d((1,1,1))
    input = torch.randn(1, 64, 10, 9, 8)
    print(input.shape)
    output = m(input)
    print(output.shape)
    output = output.view(-1, 64)
    print(output.shape)