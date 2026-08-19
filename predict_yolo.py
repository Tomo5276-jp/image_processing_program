from ultralytics import YOLO

model = YOLO("/Users/murakami/Desktop/画像処理/runs/detect/train-4/weights/best.pt")

model.predict(
    source="test_images/sample.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)

print("推論完了")