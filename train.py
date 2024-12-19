import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from timm import create_model
from torchvision import transforms, datasets
import os
import timm
import tqdm
from tqdm.auto import tqdm
# Configuration
class Config:
    model_name = "efficientformerv2_s0.snap_dist_in1k"
    num_classes = 7  # Change based on your dataset
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "/content/drive/MyDrive/assignment"
    save_path = "/content/drive/MyDrive/Sports/checkpoints/efficientformerv2_s0.pth"

# Dataset Preparation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

model = create_model(Config.model_name, pretrained=True, num_classes=Config.num_classes)
model = model.to(Config.device)

data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=True)

train_dataset = datasets.ImageFolder(os.path.join(Config.data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(Config.data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

# Model Initialization
# model = create_model(Config.model_name, pretrained=True, num_classes=Config.num_classes)
# model = model.to(Config.device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

# Training Function
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

# Validation Function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# Training Loop
best_loss = float('inf')
for epoch in range(Config.num_epochs):
    print(f"Epoch {epoch+1}/{Config.num_epochs}")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, Config.device)

    print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

    # Save the best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), Config.save_path)
        print(f"Saved Best Model with Val Loss: {best_loss:.4f}")

print("Training complete. Best model saved at", Config.save_path)
