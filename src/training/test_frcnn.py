import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# модель
model = fasterrcnn_resnet50_fpn(num_classes=5)
model.load_state_dict(torch.load("results/faster_rcnn_kitti.pth"))
model.to(device)
model.eval()

# картинка
import os

image_dir = "data/raw/training/image_2"
images = os.listdir(image_dir)[:50]
print(len(images))
print(images[:10])

for img_name in images:

    print(img_name)

    img_path = os.path.join(image_dir, img_name)

    image = Image.open(img_path).convert("RGB")

    transform = transforms.ToTensor()
    img_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    boxes = prediction[0]["boxes"].cpu()
    scores = prediction[0]["scores"].cpu()
    print(boxes)
    print(scores)

    for box, score in zip(boxes, scores):

        if score > 0.35:

            x1, y1, x2, y2 = box

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )

            ax.add_patch(rect)

    plt.savefig(f"results/frcnn/{img_name}")
    plt.close()