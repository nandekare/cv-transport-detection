import os
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# Пути
IMAGE_DIR = "data/processed/images/train"
LABEL_DIR = "data/processed/labels/train"

# Dataset
class KITTIDataset(Dataset):
    def __init__(self):
        self.images = sorted(os.listdir(IMAGE_DIR))
        self.transform = ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(IMAGE_DIR, img_name)

        label_name = img_name.replace(".png", ".txt")
        label_path = os.path.join(LABEL_DIR, label_name)

        image = Image.open(img_path).convert("RGB")

        w, h = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                cls = int(parts[0])

                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h

                bw = float(parts[3]) * w
                bh = float(parts[4]) * h

                x1 = x_center - bw / 2
                y1 = y_center - bh / 2
                x2 = x_center + bw / 2
                y2 = y_center + bh / 2

                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        image = self.transform(image)

        return image, target


# Collate function
def collate_fn(batch):
    return tuple(zip(*batch))


# Dataset
dataset = KITTIDataset()

# Loader
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn
)

# Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 5

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.0005,
    momentum=0.9,
    weight_decay=0.0005
)

# Train
model.train()

epochs = 35

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")

    for images, targets in loader:

        images = [img.to(device) for img in images]

        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device)
            }
            for t in targets
        ]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        print("Loss:", losses.item())

print("\nTraining completed!")

# Save model
torch.save(model.state_dict(), "results/faster_rcnn_kitti.pth")

print("Model saved")