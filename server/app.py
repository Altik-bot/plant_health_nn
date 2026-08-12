from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image,ImageOps
import io
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define classes
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))

state_dict = torch.load("../model/model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
target_layers = [model.layer4[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    image_resized = image.resize((224, 224))
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = predicted.item()

    # Prepare image for Grad-CAM
    rgb_image = np.array(image_resized).astype(np.float32) / 255.0

    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # Convert heatmap to bytes
    import cv2
    _, buffer = cv2.imencode(".jpg", visualization)
    heatmap_bytes = buffer.tobytes()

    return {
        "class": classes[predicted_class],
        "confidence": float(confidence.item()),
        "heatmap": heatmap_bytes.hex()
    }


