import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument(
    "mode",
    choices=["train", "test"],
    help="train or test model"
)

parser.add_argument(
    "--model",
    required=True,
    choices=[
        "yolov8",
        "yolov10",
        "frcnn",
        "ssd",
        "efficientdet"
    ]
)

args = parser.parse_args()

commands = {

    ("train", "yolov8"):
        "python src/training/train_yolov8.py",

    ("test", "yolov8"):
        "python src/training/test_yolov8.py",

    ("train", "yolov10"):
        "python src/training/train_yolov10.py",

    ("train", "frcnn"):
        "python src/training/train_frcnn_real.py",

    ("test", "frcnn"):
        "python src/training/test_frcnn.py",

    ("train", "ssd"):
        "python src/training/train_ssd.py",

    ("train", "efficientdet"):
        "python src/training/train_efficientdet.py"
}

command = commands.get((args.mode, args.model))

if command:
    print(f"Running: {command}")
    subprocess.run(command, shell=True)
else:
    print("Command not implemented")