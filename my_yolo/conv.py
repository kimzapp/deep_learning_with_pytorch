import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    Standard convolutional layer with batch normalization and activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, g=1, d=1, act: str | None ='silu'):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'silu':
            self.act = nn.SiLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {act}")


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

