import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from torch.utils.data import DataLoader
model = models.mobilenet_v2(weights="IMAGENET1K_V1")




train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("data/train", transform=train_transform)
val_data = datasets.ImageFolder("data/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(train_data.classes)
latency = []
arr_accuracy = []
num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
for param in model.parameters():
    param.requires_grad = True
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
    arr_accuracy.append(train_accuracy)
    torch.save(model.state_dict(), "mobile_model.pth")

    print(f"Epoch {epoch} done, Loss: {running_loss}, Train accuracy: {train_accuracy}")
scheduler.step()
model.eval()
all_preds = []
all_labels = []

correct = 0
total = 0
val_loss = 0
with torch.no_grad():
    for images, labels in val_loader:
        start = time.time()
        outputs = model(images)
        end  = time.time()
        latency.append(end - start)
        print(f"Latency: {end - start}s")
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
torch.save(model.state_dict(), "model.pth")     
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.title("Accuracy")   
plt.plot(arr_accuracy)
plt.show
plt.title("Latency")
plt.plot(latency)
plt.show()
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
