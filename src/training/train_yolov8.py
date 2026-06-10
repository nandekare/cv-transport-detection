from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="configs/kitti.yaml",
    epochs=40,
    imgsz=640,
    batch=8,
    workers=0,
    device=0,
    project="results",
    name="yolov8_kitti"
)
