import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# デバイス設定

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("使用デバイス:", device)

# クラス設定

classes = [
    "building",
    "facility",
    "floor"
]

# ResNet18

model = models.resnet18(weights=None)
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(classes)
)

# 学習済みモデル読み込み

model.load_state_dict(
    torch.load(
        "models/resnet_region_classifier.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# 前処理

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# 判定する画像の読み込み処理

image_path = "sample.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# 予測

with torch.no_grad():

    output = model(image)

    _, predicted = torch.max(output,1)

print("予測結果:", classes[predicted.item()])