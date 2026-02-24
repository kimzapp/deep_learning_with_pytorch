import torch
import torch.nn as nn
from block import SPPF, C2PSA


class v26Neck(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for neck layers (e.g., FPN, PAN, etc.)
        self.sppf = SPPF(1024, 1024, 5, 3, True)
        self.c2psa = C2PSA(1024, 1024, n=2, e=0.5)

    def forward(self, p3, p4, p5):
        # Placeholder for neck forward pass
        # In a real implementation, you would combine p3, p4, and p5 using FPN/PAN logic
        # For demonstration, we'll just pass p5 through SPPF and C2PSA
        x = self.sppf(p5)
        return self.c2psa(x)


if __name__ == "__main__":
    model = v26Neck()
    p3 = torch.randn(1, 256, 80, 80)  # Example input from backbone
    p4 = torch.randn(1, 512, 40, 40)  # Example input from backbone
    p5 = torch.randn(1, 1024, 20, 20)  # Example input from backbone
    output = model(p3, p4, p5)
    print(output.shape)