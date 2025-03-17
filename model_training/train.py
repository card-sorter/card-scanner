from ultralytics import YOLO

model = YOLO("../runs/segment/train8/weights/last.pt")

if __name__ == "__main__":
    results = model.train(data="cards.yaml", epochs=50, imgsz=640, save_period=1, resume=True)
    model.val()