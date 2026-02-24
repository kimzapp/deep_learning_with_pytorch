import torch
import torch.nn as nn
from conv import Conv
from block import C2f, C3k2, SPPF, C2PSA
import math


class Backbone(nn.Module):
    def __init__(self, d=1.0, w=1.0, r=1.0):
        super().__init__()
        self.p3 = nn.Identity()
        self.p4 = nn.Identity()
        self.p5 = nn.Identity()

    def forward(self, x):
        p3 = self.p3(x)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class v8Backbone(Backbone):
    def __init__(self, d=1.0, w=1.0, r=1.0):
        super().__init__()
        self.p3 = nn.Sequential(
            Conv(3, math.ceil(64*w), 3, 2, 1),
            Conv(math.ceil(64*w), math.ceil(128*w), 3, 2, 1),
            C2f(math.ceil(128*w), math.ceil(128*w), n=math.ceil(3*d), shortcut=True, e=0.25),
            Conv(math.ceil(128*w), math.ceil(256*w), 3, 2, 1),
            C2f(math.ceil(256*w), math.ceil(256*w), n=math.ceil(6*d), shortcut=True, e=0.25),
        )
        self.p4 = nn.Sequential(
            Conv(math.ceil(256*w), math.ceil(512*w), 3, 2, 1),
            C2f(math.ceil(512*w), math.ceil(512*w), n=math.ceil(6*d), shortcut=True, e=0.25),
        )
        self.p5 = nn.Sequential(
            Conv(math.ceil(512*w), math.ceil(512*w*r), 3, 2, 1),
            C2f(math.ceil(512*w*r), math.ceil(512*w*r), n=math.ceil(3*d), shortcut=True, e=0.25),
            SPPF(math.ceil(512*w*r), math.ceil(512*w*r), 5)
        )


class v26Backbone(Backbone):
    """Backbone network for YOLOv26."""

    def __init__(self, d=1.0, w=1.0, mc=512):
        """Initialize the backbone network."""
        super().__init__()
        self.p3 = nn.Sequential(
            Conv(3, math.ceil(min(64, mc)*w), 3, 2, 1),
            Conv(math.ceil(min(64, mc)*w), math.ceil(min(128, mc)*w), 3, 2, 1),
            C3k2(math.ceil(min(128, mc)*w), math.ceil(min(256, mc)*w), n=math.ceil(2*d), shortcut=False, e=0.25, c3k=False),
            Conv(math.ceil(min(256, mc)*w), math.ceil(min(256, mc)*w), 3, 2, 1),
            C3k2(math.ceil(min(256, mc)*w), math.ceil(min(512, mc)*w), n=math.ceil(2*d), shortcut=False, e=0.25, c3k=False),
        )
        self.p4 = nn.Sequential(
            Conv(math.ceil(min(512, mc)*w), math.ceil(min(512, mc)*w), 3, 2, 1),
            C3k2(math.ceil(min(512, mc)*w), math.ceil(min(512, mc)*w), n=math.ceil(2*d), shortcut=False, c3k=True),
        )
        self.p5 = nn.Sequential(
            Conv(math.ceil(min(512, mc)*w), math.ceil(max(1024, mc)*w), 3, 2, 1),
            C3k2(math.ceil(max(1024, mc)*w), math.ceil(max(1024, mc)*w), n=math.ceil(2*d), shortcut=False, c3k=True),
            SPPF(math.ceil(max(1024, mc)*w), math.ceil(max(1024, mc)*w), 5, 3, True),
            C2PSA(math.ceil(max(1024, mc)*w), math.ceil(max(1024, mc)*w), n=math.ceil(2*d), e=0.5)
        )
    

if __name__ == "__main__":
    model = v8Backbone()
    x = torch.randn(1, 3, 640, 640)  # Example input
    output = model(x)
    for o in output:
        print(o.shape)