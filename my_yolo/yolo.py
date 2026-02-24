import torch
import torch.nn as nn
from backbone import v8Backbone
from neck import v8Neck
from head import v8Head


class YOLOv8(nn.Module):
    def __init__(self, num_classes=80, d=1.0, w=1.0, r=1.0):
        super().__init__()
        self.backbone = v8Backbone(d, w, r)
        self.neck = v8Neck(d, w, r)
        self.head = v8Head(num_classes, d, w, r)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        x1, x2, x3 = self.neck(p3, p4, p5)
        locs, confs, feats = self.head(x1, x2, x3)
        return {
            'boxes': locs,
            'confs': confs,
            'feats': feats
        }
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    

if __name__ == "__main__":
    model = YOLOv8() # yolov8 l by default
    x = torch.randn(1, 3, 640, 640)  # Example input
    result = model(x)
    print("Boxes shape:", result['boxes'].shape)
    print("Confs shape:", result['confs'].shape)