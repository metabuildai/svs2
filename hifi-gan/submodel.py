import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=[1, 3, 5]):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=(kernel_size * dilation[0] - dilation[0]) // 2),
            nn.LeakyReLU(LRELU_SLOPE),
            Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)
        )
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=(kernel_size * dilation[1] - dilation[1]) // 2),
            nn.LeakyReLU(LRELU_SLOPE),
            Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=(kernel_size * dilation[2] - dilation[2]) // 2),
            nn.LeakyReLU(LRELU_SLOPE),
            Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)
        )

    def forward(self, x):
        x = self.conv0(x) + x
        x = self.conv1(x) + x
        x = self.conv2(x) + x

        return x


class ResblockSequence(torch.nn.Module):
    def __init__(self, in_channel, out_channel, conv_kernel, conv_stride, res_kernel, res_dilation):
        super(ResblockSequence, self).__init__()
        self.num_kernel = len(res_kernel)
        self.relu = nn.LeakyReLU(LRELU_SLOPE)
        self.conv = ConvTranspose1d(in_channel, out_channel, conv_kernel, conv_stride, padding=(conv_kernel-conv_stride)//2)
        self.res1 = ResBlock(out_channel, res_kernel[0], res_dilation[0])
        self.res2 = ResBlock(out_channel, res_kernel[1], res_dilation[1])
        self.res3 = ResBlock(out_channel, res_kernel[2], res_dilation[2])

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.res1(x) + self.res2(x) + self.res3(x)
        x = x / self.num_kernel

        return x


class PeriodConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(PeriodConv, self).__init__()
        self.conv = Conv2d(in_channel, out_channel, kernel, stride, padding)
        self.relu = nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.conv0 = self.period_block(1, 32, (kernel_size, 1), (stride, 1), (2, 0))
        self.conv1 = self.period_block(32, 128, (kernel_size, 1), (stride, 1), (2, 0))
        self.conv2 = self.period_block(128, 512, (kernel_size, 1), (stride, 1), (2, 0))
        self.conv3 = self.period_block(512, 1024, (kernel_size, 1), (stride, 1), (2, 0))
        self.conv4 = self.period_block(1024, 1024, (kernel_size, 1), (stride, 1), (2, 0))
        self.conv_post = nn.Sequential(
            Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)),
            nn.Sigmoid()
        )

    def period_block(self, in_channel, out_channel, kernel, stride, padding):
        return torch.nn.Sequential(
            Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding),
            nn.LeakyReLU(LRELU_SLOPE)
        )

    def forward(self, x):
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.conv_post(x4).flatten()

        return x, [x0, x1, x2, x3, x4]


class DiscriminatorS(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        self.conv0 = self.scale_block(1, 128, 15, 1, 7, 1)
        self.conv1 = self.scale_block(128, 128, 41, 2, 20, 4)
        self.conv2 = self.scale_block(128, 256, 41, 2, 20, 16)
        self.conv3 = self.scale_block(256, 512, 41, 4, 20, 16)
        self.conv4 = self.scale_block(512, 1024, 41, 4, 20, 16)
        self.conv5 = self.scale_block(1024, 1024, 41, 1, 20, 16)
        self.conv6 = self.scale_block(1024, 1024, 5, 1, 2, 1)
        self.conv_post = nn.Sequential(
            Conv1d(1024, 1, 3, 1, padding=1),
            nn.Sigmoid()
        )

    def scale_block(self, in_channel, out_channel, kernel_size, stride, padding, groups):
        return torch.nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, groups=groups),
            nn.LeakyReLU(LRELU_SLOPE)
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x = self.conv_post(x6).flatten()

        return x, [x0, x1, x2, x3, x4, x5, x6]
