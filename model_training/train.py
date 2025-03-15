from ultralytics import YOLO

model = YOLO("yolo11n-seg.yaml")

if __name__ == "__main__":
    results = model.train(data="cards.yaml", epochs=100, imgsz=640)
    model.val()