from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.predict(
    source="data/test50",
    save=True,
    show=True,
    project="results",
    name="yolov10_50",
    conf=0.3
)