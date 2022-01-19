import torch
from torch.nn import AvgPool1d
from submodel import DiscriminatorP, DiscriminatorS


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.dis0 = DiscriminatorP(2)
        self.dis1 = DiscriminatorP(3)
        self.dis2 = DiscriminatorP(5)
        self.dis3 = DiscriminatorP(7)
        self.dis4 = DiscriminatorP(11)

    def forward(self, x):
        x0, feature0 = self.dis0(x)
        x1, feature1 = self.dis1(x)
        x2, feature2 = self.dis2(x)
        x3, feature3 = self.dis3(x)
        x4, feature4 = self.dis4(x)

        return [x0, x1, x2, x3, x4], feature0 + feature1 + feature2 + feature3 + feature4


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.dis0 = DiscriminatorS()
        self.dis1 = DiscriminatorS()
        self.dis2 = DiscriminatorS()
        self.pool0 = AvgPool1d(4, 2, padding=2)
        self.pool1 = AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        x0, feature0 = self.dis0(x)
        x = self.pool0(x)
        x1, feature1 = self.dis1(x)
        x = self.pool1(x)
        x2, feature2 = self.dis1(x)

        return [x0, x1, x2], feature0 + feature1 + feature2