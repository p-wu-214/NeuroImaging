import torch.nn as nn


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()]
    ])[activation]


def conv3d(in_ch, out_ch, stride=1, padding=1):
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=padding, bias=False)


def conv_block(in_ch, out_ch, expansion, stride=1):
    out_ch = out_ch * expansion
    return nn.Sequential(
        nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride*expansion,
                  padding=(1, 1, 1), groups=1),
        nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride,
                  padding=(1, 1, 1), groups=1),
        nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride,
                  padding=(1, 1, 1), groups=1),
        nn.BatchNorm3d(out_ch),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, stride=stride, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = conv3d(out_ch, out_ch)
        self.conv3 = conv3d(out_ch, out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, stride=stride, kernel_size=1)
        self.batch = nn.BatchNorm3d(out_ch)
        self.relu = activation_func('relu')

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch(x)
        residual = self.shortcut(residual)
        x += residual
        x = self.relu(x)
        return x


class ExpandBlock(nn.Module):
    expansion = 2
    stride = 1

    def __init__(self, in_ch, out_ch):
        super(ExpandBlock, self).__init__()
        self.block = conv_block(in_ch, out_ch, expansion=2)
        self.relu = activation_func('relu')
        self.shortcut = nn.Conv3d(in_ch, out_ch * self.expansion, kernel_size=1, stride=self.stride * self.expansion)

    def forward(self, x):
        residual = x
        x = self.block(x)
        residual = self.shortcut(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNetBasicBlock(nn.Module):

    def __init__(self, batch_size):
        super(ResNetBasicBlock, self).__init__()
        self.batch_size = batch_size
        self.basic_block1 = BasicBlock(53, 64)
        self.expand_block1 = ExpandBlock(64, 64)
        self.expand_block2 = ExpandBlock(128, 128)
        self.average_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(256, out_features=5, bias=True)

    def forward(self, x):
        x = self.basic_block1(x)
        x = self.expand_block1(x)
        x = self.expand_block2(x)
        x = self.average_pool(x)
        x = x.view(-1, 256)
        x = self.linear(x)

        return x
