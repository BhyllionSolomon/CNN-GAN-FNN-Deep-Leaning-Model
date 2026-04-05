import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError

# ✅ Dataset paths
base_dir = r"C:\Users\Solomon\Documents\CNN_GANS_LSTM\TomatoClass_Split"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# ✅ Function to clean corrupted images
def clean_corrupted_images(folder_path):
    removed = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            try:
                Image.open(img_path).convert("RGB")  # Try opening the image
            except UnidentifiedImageError:
                print(f"⚠️ Removing corrupted image: {img_path}")
                os.remove(img_path)
                removed += 1
    if removed > 0:
        print(f"✅ Cleaned {removed} corrupted images from {folder_path}")
    else:
        print(f"✅ No corrupted images found in {folder_path}")

# ✅ Clean corrupted images from train & val datasets
clean_corrupted_images(train_dir)
clean_corrupted_images(val_dir)

# ✅ Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Load datasets (safe now)
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

# ✅ Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define CNN model (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 🔹 Two classes: Ripe & Occluded
model = model.to(device)

# ✅ Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ✅ Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}] "
          f"Loss: {running_loss / len(train_loader):.4f} | "
          f"Accuracy: {train_acc:.2f}%")

# ✅ Save trained model
model_path = os.path.join(base_dir, "tomato_cnn.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved successfully: {model_path}")
