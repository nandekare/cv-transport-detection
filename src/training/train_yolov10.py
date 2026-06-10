from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(
    data="configs/kitti.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    workers=0,
    device=0,
    project="results",
    name="yolov10_kitti"
)