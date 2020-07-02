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
        self.basic_block1 = BasicBlock(64, 64)
        self.expand_block1 = ExpandBlock(64, 64)
        self.basic_block2 = BasicBlock(128, 128)
        self.expand_block2 = ExpandBlock(128, 128)
        self.basic_block3 = BasicBlock(256, 256)
        self.expand_block3 = ExpandBlock(256, 256)
        self.basic_block4 = BasicBlock(512, 512)
        self.average_pool = nn.AdaptiveAvgPool3d((30, 30, 30))
        self.linear = nn.Linear(30*30*30*512, out_features=5, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.basic_block1(x)
        x = self.expand_block1(x)
        x = self.basic_block2(x)
        x = self.expand_block2(x)
        x = self.basic_block3(x)
        x = self.expand_block3(x)
        x = self.basic_block4(x)
        x = self.average_pool(x)
        x = x.view(config.hyper_params['batch'], -1)
        x = self.linear(x)
        x = self.relu(x)

        return x
