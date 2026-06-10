import shutil
import random
from pathlib import Path

# Пути

BASE_DIR = Path("data/raw/training")
IMAGE_DIR = BASE_DIR / "image_2"
LABEL_DIR = BASE_DIR / "label_2"

OUTPUT_DIR = Path("data/processed")

# Классы

CLASSES = ["Car"]

# Создаем папки

for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Получаем список изображений

image_files = list(IMAGE_DIR.glob("*.png"))
random.shuffle(image_files)

# Делим train/val

split_index = int(len(image_files) * 0.8)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

def convert_label(label_path, output_path, img_width=1242, img_height=375):
    lines_out = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        cls = parts[0]

        if cls not in CLASSES:
            continue

        class_id = CLASSES.index(cls)

        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])

        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height

        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        lines_out.append(
            f"{class_id} {x_center} {y_center} {width} {height}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines_out))

def process_split(files, split_name):
    for img_path in files:
        label_path = LABEL_DIR / (img_path.stem + ".txt")

        out_img = OUTPUT_DIR / "images" / split_name / img_path.name
        out_label = OUTPUT_DIR / "labels" / split_name / (img_path.stem + ".txt")

        shutil.copy(img_path, out_img)

        convert_label(label_path, out_label)

process_split(train_files, "train")
process_split(val_files, "val")

print("KITTI successfully converted to YOLO format!")
