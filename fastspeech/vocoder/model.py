import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
        )
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
            nn.LeakyReLU(LRELU_SLOPE),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
        )

    def forward(self, x):
        x = self.conv0(x) + x
        x = self.conv1(x) + x
        x = self.conv2(x) + x

        return x

    def remove_weight_norm(self):
        for i, l in enumerate(self.conv0):
            if i % 2 == 1:
                remove_weight_norm(l)
        for i, l in enumerate(self.conv1):
            if i % 2 == 1:
                remove_weight_norm(l)
        for i, l in enumerate(self.conv2):
            if i % 2 == 1:
                remove_weight_norm(l)


class ResblockSequence(torch.nn.Module):
    def __init__(self, in_channel, out_channel, conv_kernel, conv_stride, res_kernel, res_dilation):
        super(ResblockSequence, self).__init__()
        self.num_kernel = len(res_kernel)
        self.relu = nn.LeakyReLU(LRELU_SLOPE)
        self.conv = weight_norm(
                ConvTranspose1d(in_channel, out_channel, conv_kernel, conv_stride, padding=(conv_kernel-conv_stride)//2))
        self.res1 = ResBlock(out_channel, res_kernel[0], res_dilation[0])
        self.res2 = ResBlock(out_channel, res_kernel[1], res_dilation[1])
        self.res3 = ResBlock(out_channel, res_kernel[2], res_dilation[2])

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.res1(x) + self.res2(x) + self.res3(x)
        x = x / self.num_kernel

        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)
        self.res1.remove_weight_norm()
        self.res2.remove_weight_norm()
        self.res3.remove_weight_norm()


class Generator(torch.nn.Module):
    def __init__(self, resblock_kernel_size, upsample_rate, upsample_initial_channel, upsample_kernel_size,
                 mode_resblock, resblock_dilation_size, num_mel):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_size)
        self.num_upsamples = len(upsample_rate)
        self.conv_pre = weight_norm(Conv1d(num_mel, upsample_initial_channel, 7, 1, padding=3))
        self.res0 = ResblockSequence(upsample_initial_channel // 1, upsample_initial_channel // (2 ** 1),
                                     upsample_kernel_size[0], upsample_rate[0],
                                     resblock_kernel_size, resblock_dilation_size)
        self.res1 = ResblockSequence(upsample_initial_channel // (2 ** 1), upsample_initial_channel // (2 ** 2),
                                     upsample_kernel_size[1], upsample_rate[1],
                                     resblock_kernel_size, resblock_dilation_size)
        self.res2 = ResblockSequence(upsample_initial_channel // (2 ** 2), upsample_initial_channel // (2 ** 3),
                                     upsample_kernel_size[2], upsample_rate[2],
                                     resblock_kernel_size, resblock_dilation_size)
        self.res3 = ResblockSequence(upsample_initial_channel // (2 ** 3), upsample_initial_channel // (2 ** 4),
                                     upsample_kernel_size[3], upsample_rate[3],
                                     resblock_kernel_size, resblock_dilation_size)

        self.conv_post = weight_norm(Conv1d(upsample_initial_channel // (2 ** 4), 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)
        self.res0.remove_weight_norm()
        self.res1.remove_weight_norm()
        self.res2.remove_weight_norm()
        self.res3.remove_weight_norm()
        remove_weight_norm(self.conv_post)
