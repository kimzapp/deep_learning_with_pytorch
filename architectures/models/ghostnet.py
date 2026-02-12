import torch
import torch.nn as nn
import torch.nn.functional as F

# se module
class SEModule(nn.Module):
    def __init__(self, input_channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(input_channel, input_channel // reduction),
            nn.ReLU(),
            nn.Linear(input_channel // reduction, input_channel)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)
        y = self.layers(y).reshape(b, c, 1, 1)
        y = F.sigmoid(y)
        return x * y

# ghost module

# ghost bottleneck

if __name__ == "__main__":
    se_module = SEModule(input_channel=16, reduction=4)
    test_input = torch.randn(4, 16, 224, 224)
    output = se_module(test_input)
    print(output.shape)