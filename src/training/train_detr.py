from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="configs/kitti.yaml",
    epochs=10,
    imgsz=640,
    batch=2,
    workers=0,
    device=0,
    project="results",
    name="detr_kitti"
)