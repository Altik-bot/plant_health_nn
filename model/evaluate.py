import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# Model
model = models.resnet18(weights=None)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_data = datasets.ImageFolder(
    "data/val",
    transform=val_transform
)

val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(val_data.classes)
)

model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)

model.eval()


all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Classes:")
print(val_data.classes)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=val_data.classes
    )
)