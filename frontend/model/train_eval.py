import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
model = models.resnet18(pretrained=True)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("data/train", transform=transform)
val_data = datasets.ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(train_data.classes)

num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss  += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_accuracy = 100*( correct / total ) 
    torch.save(model.state_dict(), "model.pth")

    print(f"Epoch {epoch} done, Loss: {running_loss}, Train accuracy: {train_accuracy}")
model.eval()
correct = 0
total = 0


with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        running_loss  += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_accuracy = 100*( correct / total ) 
        
print("Accuracy:",val_accuracy)
