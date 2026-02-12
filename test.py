from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
print(type(model.model))

test_input = torch.randn(1, 3, 640, 640)  # Example input tensor (batch_size, channels, height, width)
predictions = model.model(test_input)  # Get predictions from the model
print(predictions[1]['scores'].shape)

for param in model.model.parameters():
    print(param.shape)