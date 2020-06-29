import torch.nn as nn
import torch_xla.core.xla_model as xm
import config

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()]
    ])[activation]


def conv3d(in_ch, out_ch, stride=1, padding=1):
    return nn.Conv3d(in_ch, out_ch, stride=stride, kernel_size=3, padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3d(out_ch, out_ch)
        self.conv2 = conv3d(out_ch, out_ch)
        self.conv3 = conv3d(out_ch, out_ch)
        self.relu = activation_func('relu')
        self.batch = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        x = self.relu(x)
        x = self.batch(x)
        return x


class ExpandBlock(nn.Module):
    expansion = 2
    stride = 1

    def __init__(self, in_ch, out_ch, stride=1, expansion=2):
        super(ExpandBlock, self).__init__()
        out_ch = out_ch*expansion
        self.conv1 = conv3d(in_ch, out_ch, stride=stride*expansion, padding=1)
        self.conv2 = conv3d(out_ch, out_ch, padding=1)
        self.conv3 = conv3d(out_ch, out_ch, padding=1)
        self.relu = activation_func('relu')
        self.batch = nn.BatchNorm3d(out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride * expansion)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.shortcut(residual)
        x += residual
        x = self.relu(x)
        x = self.batch(x)
        return x


class ResNetBasicBlock(nn.Module):

    def __init__(self):
        super(ResNetBasicBlock, self).__init__()
        self.input_conv = nn.Conv3d(53, 64, stride=(1, 1, 1), kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.input_batch = nn.BatchNorm3d(64)
        self.input_relu = activation_func('relu')
        self.input_max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.basic_block1 = BasicBlock(64, 64)
        self.expand_block1 = ExpandBlock(64, 64)
        self.basic_block2 = BasicBlock(128, 128)
        self.expand_block2 = ExpandBlock(128, 128)
        self.average_pool = nn.AdaptiveAvgPool3d((15, 15, 15))
        self.linear = nn.Linear(15*15*15*256, out_features=5, bias=True)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.basic_block1(x)
        x = self.expand_block1(x)
        x = self.basic_block2(x)
        x = self.expand_block2(x)
        x = self.average_pool(x)
        x = x.view(config.hyper_params['batch'], -1)
        x = self.linear(x)

        return x
