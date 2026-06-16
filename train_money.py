from ultralytics import YOLO

if __name__ == "__main__":
    # For a new model, use:
    # model = YOLO("yolo26n.pt")

    model = YOLO(r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\runs\detect\train-26\weights\best.pt")
    model.train(
        data=r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\currency\data.yaml",
        epochs=15,
        imgsz=640,
        device=0,
        workers=0
    )