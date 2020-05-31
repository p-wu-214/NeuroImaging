import torch.nn as nn
import torch_xla.core.xla_model as xm


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()]
    ])[activation]


def conv3d(in_ch, out_ch, stride=1, padding=1):
    return nn.Conv3d(in_ch, out_ch, stride=stride, kernel_size=3, padding=padding, bias=False)


# def conv_block(in_ch, out_ch, expansion=1, stride=1):
#     out_ch = out_ch * expansion
#     return nn.Sequential(
#         nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride*expansion,
#                   padding=(1, 1, 1), groups=1),
#         nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride,
#                   padding=(1, 1, 1), groups=1),
#         nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=stride,
#                   padding=(1, 1, 1), groups=1),
#         nn.BatchNorm3d(out_ch),
#     )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3d(out_ch, out_ch)
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

    def __init__(self, in_ch, out_ch, stride=1, expansion=2):
        super(ExpandBlock, self).__init__()
        out_ch = out_ch*expansion
        self.conv1 = conv3d(in_ch, out_ch, stride=stride*expansion, padding=1)
        self.conv2 = conv3d(out_ch, out_ch, padding=1)
        self.conv3 = conv3d(out_ch, out_ch, padding=1)
        self.batch = nn.BatchNorm3d(out_ch)
        self.relu = activation_func('relu')
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride * expansion)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.shortcut(residual)
        xm.master_print(f'x: {x.shape} residual: {residual.shape}')
        x += residual
        x = self.batch(x);
        x = self.relu(x)
        return x


class ResNetBasicBlock(nn.Module):

    def __init__(self, batch_size):
        super(ResNetBasicBlock, self).__init__()
        self.batch_size = batch_size
        self.input_conv = nn.Conv3d(53, 64, stride=(1, 1, 1), kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.input_batch = nn.BatchNorm3d(64)
        self.input_relu = activation_func('relu')
        self.input_max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.basic_block1 = BasicBlock(64, 64)
        self.expand_block1 = ExpandBlock(64, 64)
        self.basic_block2 = BasicBlock(128, 128)
        self.expand_block2 = ExpandBlock(128, 128)
        self.average_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(256, out_features=5, bias=True)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.basic_block1(x)
        x = self.expand_block1(x)
        x = self.basic_block2(x)
        x = self.expand_block2(x)
        print('Before average pool: ' + str(x.shape))
        x = self.average_pool(x)
        print('After average pool: ' + str(x.shape))
        x = x.view(-1, 256)
        print('After view: ' + str(x.shape))
        x = self.linear(x)
        print('After linear: ' + str(x.shape))

        return x
