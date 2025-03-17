from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../runs/segment/train8/weights/last.pt")

results = model(source=2, show=True)