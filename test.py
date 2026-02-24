import torch
from ultralytics import YOLO

model = YOLO("yolo26l.pt")
net = model.model
net.eval()

x = torch.randn(1, 3, 640, 640)

features = []
strides = []

def hook_fn(module, input, output):
    features.append(output)
    strides.append(640 // output.shape[-1])

hooks = []

# Hook tất cả layer backbone
for layer in net.model:
    hooks.append(layer.register_forward_hook(hook_fn))

_ = net(x)

# Lọc P3, P4, P5 theo stride
P3 = None
P4 = None
P5 = None

for feat, s in zip(features, strides):
    if s == 8:
        P3 = feat
    elif s == 16:
        P4 = feat
    elif s == 32:
        P5 = feat

print("P3:", P3.shape)
print("P4:", P4.shape)
print("P5:", P5.shape)

for h in hooks:
    h.remove()