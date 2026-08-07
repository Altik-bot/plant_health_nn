import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



classes = { 
    0: 'Scab',
    1: 'Black_rot',
    2: 'Rust',
    3: 'Healthy'
    }

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


image_path = "Healthy2.JPG"

image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

rgb_image = torch.tensor(
    __import__("numpy").array(image)
).float() / 255.0

input_tensor = transform(image).unsqueeze(0)


with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = outputs.argmax(dim=1).item()

print("Prediction:", classes[predicted_class])


target_layers = [model.layer4[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [ClassifierOutputTarget(predicted_class)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

visualization = show_cam_on_image(
    rgb_image.numpy(),
    grayscale_cam,
    use_rgb=True
)


plt.figure(figsize=(8, 8))
plt.imshow(visualization)
plt.axis("off")
plt.title(classes[predicted_class])
plt.show()