import torch
import torch.nn as nn


class ResNextBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNextBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1d_1x1=nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.out_channel),

        )
        self.conv1d_3x3 = nn.Sequential(
            nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm1d(self.out_channel),

        )
        self.conv1d_1x1_up=nn.Sequential(
            nn.Conv1d(in_channels=self.out_channel, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channel),
        )
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x1=self.conv1d_1x1(x)
        x2=self.conv1d_3x3(x1)
        x3=self.conv1d_1x1_up(x2)

        return x3+x

if __name__ == '__main__':
    x = torch.randn(64, 256, 1)
    out = ResNextBlock(256, 128)(x)
    print(out.shape)
