import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# デバイス設定

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("使用デバイス:", device)

# データセット

data_dir = "dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

print("クラス:", dataset.classes)

# ResNet

model = models.resnet18(weights=None)

# 出力を3クラスへ変更
num_features = model.fc.in_features
model.fc = nn.Linear(
    num_features,
    len(dataset.classes)
)

model = model.to(device)

# 損失関数・最適化

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# 学習

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss:{running_loss:.4f} "
        f"Accuracy:{accuracy:.2f}%"
    )

# 保存

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/resnet_region_classifier.pth"
)

print("学習完了")
print("models/resnet_region_classifier.pth を保存しました")