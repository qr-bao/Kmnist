import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
from typing import List

# using the same model architecture as the training script 
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class PreActResNet18(nn.Module):
    def __init__(self, num_classes=49):
        super(PreActResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(in_planes, out_planes, stride))
            in_planes = out_planes
            stride = 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.linear(out)

# load the trained model
def load_model():
    model = PreActResNet18(num_classes=49)
    model.load_state_dict(torch.load('./checkpoints/best_model1.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化 FastAPI 应用
app = FastAPI()

class PredictionResponse(BaseModel):
    predictions: List[int]

# image preprocessing function
def preprocess_image(image: Image.Image):
    image = image.convert("L").resize((28, 28))  # transform to grayscale and resize
    img_array = np.array(image) / 255.0  #  scale to [0, 1]
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    return img_tensor

# single image prediction endpoint
async def predict_single_image(image: Image.Image):
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# batch prediction endpoint
@app.post("/predict_batch", response_model=PredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    predictions = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read()))
        prediction = await predict_single_image(image)
        predictions.append(prediction)
    return {"predictions": predictions}
