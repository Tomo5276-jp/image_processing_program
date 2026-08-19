from ultralytics import YOLO

def main():

    model = YOLO("yolov8n.pt")

    model.train(
        data="Object-Detection-5/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu"
    )

if __name__ == "__main__":
    main()