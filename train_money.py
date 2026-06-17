from ultralytics import YOLO

if __name__ == "__main__":
    # For a new model, use:
    # model = YOLO("yolo26n.pt")

    model = YOLO(r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\runs\detect\train-53 epochs\weights\best.pt")
    model.train(
        data=r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\data.yaml",
        epochs=97,
        imgsz=640,
        device=0,
        workers=0
    )