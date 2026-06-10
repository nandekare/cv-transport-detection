import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Загружаем SSD модель
model = ssd300_vgg16(pretrained=True)
model.eval()

print("SSD model loaded")


import os

image_dir = "data/raw/training/image_2"
images = os.listdir(image_dir)[:50]

for img_name in images:

    print(img_name)

    img_path = os.path.join(image_dir, img_name)

    # Загрузка изображения
    image = Image.open(img_path).convert("RGB")

    transform = transforms.ToTensor()
    img_tensor = transform(image)

    # Предсказание
    with torch.no_grad():
        predictions = model([img_tensor])

    print(predictions)

    # Отрисовка
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    print(boxes)
    print(scores)

    for i in range(len(boxes)):
        if scores[i] > 0.55:
            box = boxes[i]

            x1, y1, x2, y2 = box

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='blue',
                facecolor='none'
            )

            ax.add_patch(rect)

            plt.text(
                x1,
                y1,
                f"{scores[i]:.2f}",
                color='blue',
                fontsize=10,
                bbox=dict(facecolor='white')
            )

    plt.savefig(f"results/ssd/{img_name}")
    plt.close()

    print("SSD result saved")