from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\chest\OneDrive\Documents\YOLOProjects\runs\detect\train-23\weights\best.pt")

    model.train(
        data="C:/Users/chest/OneDrive/Documents/YOLOProjects/currency/data.yaml",
        epochs=44,
        imgsz=640,
        device=0,
        workers=0
    )