import torch
import torch.nn as nn
from typing import Tuple
from conv import Conv
from math import ceil


class v8Head(nn.Module):
    def __init__(self, num_classes=80, d=1.0, w=1.0, r=1.0, reg_max=16):
        super().__init__()
        # Placeholder for head layers (e.g., detection layers, etc.)
        self.reg_max = reg_max
        self.num_classes = num_classes

        in_chs = [ceil(256*w), ceil(512*w), ceil(512*w*r)]
        c2, c3 = max((16, in_chs[0] // 4, reg_max * 4)), max(in_chs[0], min(num_classes, 100))  # channels
        self.localize = nn.ModuleList([
            nn.Sequential(
                Conv(x, c2, 3, 1, 1),
                Conv(c2, c2, 1, 1, 0),
                nn.Conv2d(c2, 4 * reg_max, 1)
            ) for x in in_chs
        ])
        self.classify = nn.ModuleList([
            nn.Sequential(
                Conv(x, c3, 3, 1, 1),
                Conv(c3, c3, 1, 1, 0),
                nn.Conv2d(c3, num_classes, 1)
            ) for x in in_chs
        ])

    def forward(self, x1, x2, x3):
        b, c, h, w = x1.shape
        locs = [l(x).reshape(b, 4 * self.reg_max, -1) for l, x in zip(self.localize, [x1, x2, x3])]
        confs = [c(x).reshape(b, self.num_classes, -1) for c, x in zip(self.classify, [x1, x2, x3])]
        return torch.cat(locs, dim=2), torch.cat(confs, dim=2), [x1, x2, x3]


if __name__ == "__main__":
    model = v8Head()
    x1 = torch.randn(1, 256, 80, 80)  # Example input from neck
    x2 = torch.randn(1, 512, 40, 40)  # Example input from neck
    x3 = torch.randn(1, 512, 20, 20)  # Example input from neck
    locs, confs, _ = model(x1, x2, x3)
    print(locs.shape)  # Should be (1, 4*reg_max, num_anchors)
    print(confs.shape)  # Should be (1, num_classes,