import torch
from effdet import create_model
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Загружаем EfficientDet
model = create_model(
    'tf_efficientdet_d0',
    bench_task='predict',
    pretrained=True
)

model.eval()

print("EfficientDet loaded")

# Картинка
import os

image_dir = "data/raw/training/image_2"
images = os.listdir(image_dir)[:50]

for img_name in images:

    print(img_name)

    img_path = os.path.join(image_dir, img_name)

    image = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        predictions = model(img_tensor)

    print(predictions)

    # Визуализация
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # predictions shape
    preds = predictions[0]

    for pred in preds:
        print(pred)
        score = pred[4]

        if score > 0.3:
            orig_w, orig_h = image.size

            x_scale = orig_w / 512
            y_scale = orig_h / 512

            x1, y1, x2, y2 = pred[:4]

            x1 *= x_scale
            x2 *= x_scale
            y1 *= y_scale
            y2 *= y_scale

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='green',
                facecolor='none'
            )

            ax.add_patch(rect)

            plt.text(
                x1,
                y1,
                f"{score:.2f}",
                color='green',
                fontsize=10,
                bbox=dict(facecolor='white')
            )

    plt.savefig(f"results/efficientdet/result_{img_name}.png")
    plt.close()
    print(f"saved {img_name}")
    print("EfficientDet result saved")