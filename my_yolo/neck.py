import torch
import torch.nn as nn
from block import SPPF, C2PSA, C2f, Conv
from math import ceil


class Neck(nn.Module):
    """Base class for neck networks."""
    def __init__(self):
        super().__init__()
        self.top_down = nn.ModuleList([
            nn.Identity() for _ in range(4)
        ])
        self.bottom_up = nn.ModuleList([
            nn.Identity() for _ in range(4)
        ])

    def forward(self, p3, p4, p5):
        x1 = x3 = p5
        x1 = self.top_down[1](torch.cat([self.top_down[0](x1), p4], dim=1))
        x2 = x1
        x1 = self.top_down[3](torch.cat([self.top_down[2](x1), p3], dim=1))

        x2 = self.bottom_up[1](torch.cat([self.bottom_up[0](x1), x2], dim=1))
        x3 = self.bottom_up[3](torch.cat([self.bottom_up[2](x2), x3], dim=1))
        return x1, x2, x3


class v8Neck(Neck):
    def __init__(self, d=1.0, w=1.0, r=1.0):
        super().__init__()
        # Placeholder for neck layers (e.g., FPN, PAN, etc.)
        self.top_down = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f(ceil(512*w*(1+r)), ceil(512*w), n=ceil(3*d), shortcut=False, e=0.25),
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f(ceil(768*w), ceil(256*w), n=ceil(3*d), shortcut=False, e=0.25),
        ])
        self.bottom_up = nn.ModuleList([
            Conv(ceil(256*w), ceil(256*w), 3, 2, 1),
            C2f(ceil(768*w), ceil(512*w), n=ceil(3*d), shortcut=False, e=0.25),
            Conv(ceil(512*w), ceil(512*w), 3, 2, 1),
            C2f(ceil(512*w*(1+r)), ceil(512*w*r), n=ceil(3*d), shortcut=False, e=0.25),
        ])


class v26Neck(Neck):
    def __init__(self):
        super().__init__()
        # Placeholder for neck layers (e.g., FPN, PAN, etc.)
        self.sppf = SPPF(1024, 1024, 5, 3, True)
        self.c2psa = C2PSA(1024, 1024, n=2, e=0.5)


if __name__ == "__main__":
    model = v8Neck()
    p3 = torch.randn(1, 256, 80, 80)  # Example input from backbone
    p4 = torch.randn(1, 512, 40, 40)  # Example input from backbone
    p5 = torch.randn(1, 512, 20, 20)  # Example input from backbone
    output = model(p3, p4, p5)
    for o in output:
        print(o.shape)