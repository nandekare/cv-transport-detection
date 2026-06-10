from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict(
    source="data/test50",
    save=True,
    project="results",
    name="yolov8_50",
    conf=0.25,
    show=True
)