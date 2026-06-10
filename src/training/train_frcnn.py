import torch
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Загружаем модель
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

print("Model loaded")

# Загружаем изображение
img_path = "data/processed/images/train/002759.png"

image = Image.open(img_path).convert("RGB")

transform = transforms.ToTensor()
img_tensor = transform(image)

# Предсказание
with torch.no_grad():
    predictions = model([img_tensor])

print(predictions)

# Рисуем результат
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
    if score > 0.35:
        x1, y1, x2, y2 = box.numpy()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )

        ax.add_patch(rect)

plt.show()