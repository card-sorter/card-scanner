from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../runs/segment/train9/weights/best.pt")

results = model(source=2, show=True, stream=True)
for result in results:
    print(result)