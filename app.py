from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io
import torchvision.models as models
import torch.nn as nn
import torch

model = models.resnet18(pretrained=False)
app = FastAPI()
model.fc = nn.Linear(model.fc.in_features, 4)

state_dict = torch.load("../model/model.pth", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes  = {
    0: 'Scab',
    1: 'Black_rot',
    2: 'Rust',
    3: 'Healthy'
}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "class": classes[predicted.item()],
        "confidence": float(confidence.item())
    }
