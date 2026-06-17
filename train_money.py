from ultralytics import YOLO

if __name__ == "__main__":
    # For a new model, use:
    # model = YOLO("yolo26n.pt")

    model = YOLO("yolo26s.pt")
    model.train(
        data=r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\data.yaml",
        epochs=10,
        imgsz=960,
        device=0,
        workers=0,
        scale=0.5,
        dropout=0.1,
        degrees=15.0,
        perspective=0.001,
        mixup=0.1,
        copy_paste=0.1,
        batch=8,
        pretrained=r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\runs\detect\train-150 epochs\weights\best.pt",
        patience=5,
        cache='disk'
    )