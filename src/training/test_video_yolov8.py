from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict(
    source="videos/traffic.mp4",
    save=True,
    conf=0.25,
    iou=0.5,
    classes=[2]
)
