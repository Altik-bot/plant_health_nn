import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import csv

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from torch.utils.data import DataLoader
model = models.resnet18(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),

    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),

    transforms.GaussianBlur(kernel_size=3),

    transforms.ToTensor(),normalize
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),normalize
])

train_data = datasets.ImageFolder("data/train", transform=train_transform)
val_data = datasets.ImageFolder("data/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(train_data.classes)
latency = []
arr_accuracy = []
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
num_epochs = 10
result = {}
for epoch in range(num_epochs):
    
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    if epoch == 3:
        for param in model.parameters():
            param.requires_grad = True
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
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
    result[epoch] = train_accuracy
    scheduler.step() 
    print(f"Epoch {epoch} done, Loss: {running_loss}, Train accuracy: {train_accuracy}")

model.eval()
all_preds = []
all_labels = []

correct = 0
total = 0
val_loss = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        start = time.time()
        outputs = model(images)
        end  = time.time()
        latency.append((end - start) / images.size(0))
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
plt.show()
plt.title("Latency")
plt.plot(latency)
plt.show()
result["model"] = "resnet18"
result["Accuracy"] = accuracy
result["Precision"] = precision
result["Recall"] = recall
result["F1"] = f1
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
def writedown(results):
    with open("results.csv",mode = 'w')as f :
        a = csv.writer(f,delimiter = ",")
        for k in results.keys():
            a.writerow([k,results[k]])
writedown(result)
