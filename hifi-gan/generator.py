import torch
from torch.nn import Conv1d, functional as F
from submodel import ResblockSequence


class Generator(torch.nn.Module):
    def __init__(self, resblock_kernel_size, upsample_rate, upsample_initial_channel, upsample_kernel_size,
                 resblock_dilation_size, num_mel):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_size)
        self.num_upsamples = len(upsample_rate)
        self.conv_pre = Conv1d(num_mel, upsample_initial_channel, 7, 1, padding=3)
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

        self.conv_post = Conv1d(upsample_initial_channel // (2 ** 4), 1, 7, 1, padding=3)

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
