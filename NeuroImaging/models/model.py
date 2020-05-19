import torch.nn as nn

from functools import partial


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()]
    ])[activation]


class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
            self.kernel_size[2] // 2
        )  # dynamic add padding based on the kernel_size


conv3d = partial(Conv3dAuto, kernel_size=3, bias=False)


def conv_block(in_ch, out_ch, expansion):
    stride = 1
    return nn.Sequential(
        nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride * expansion,
                  groups=1),
        nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride * expansion,
                  groups=1),
        nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride * expansion,
                  groups=1),
        nn.BatchNorm3d(out_ch),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3d(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Conv3d(out_ch, out_ch, stride=stride, kernel_size=(3,3,3))
        self.conv3 = nn.Conv3d(out_ch, out_ch, stride=stride, kernel_size=(3,3,3))
        self.batch = nn.BatchNorm3d(out_ch)
        self.relu = activation_func('relu')

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch(x)
        x += residual
        x = self.relu(x)
        return x


class ExpandBlock(nn.Module):
    expansion = 2

    def __init__(self, in_ch, out_ch):
        super(ExpandBlock, self).__init__()
        self.block = conv_block(in_ch, out_ch * 2, expansion=2)
        self.relu = activation_func('relu')

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        x = self.relu(x)
        return x


class ResNetBasicBlock(nn.Module):

    def __init__(self):
        super(ResNetBasicBlock, self).__init__()
        self.basic_block1 = BasicBlock(53, 64)
        self.expand_block1 = ExpandBlock(64, 64)
        self.expand_block2 = ExpandBlock(128, 128)
        self.expand_block3 = ExpandBlock(256, 256)
        self.linear = nn.Linear(512, 5, bias=True)

    def forward(self, x):
        x = self.basic_block1(x)
        x = self.expand_block1(x)
        x = self.expand_block2(x)
        x = self.linear(x)

        return x
