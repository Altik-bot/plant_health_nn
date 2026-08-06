import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_data = datasets.ImageFolder("data/train", transform=transform)
val_data = datasets.ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
num_classes = len(train_data.classes)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()


img = Image.open("ff0f9b3a-4ef4-4c38-9a23-516beff99192___JR_FrgE.S 3048.JPG").convert("RGB")
img = transform(img)
img = img.unsqueeze(0)
with torch.no_grad():
    outputs = model(img)
_, predicted = torch.max(outputs, 1)
predicted_class = predicted.item()
class_map = {
    0: 'Scab',
    1: 'Black_rot',
    2: 'Rust',
    3: 'Healthy'
}
print(class_map[predicted_class])
