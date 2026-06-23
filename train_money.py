from ultralytics import YOLO

if __name__ == "__main__":
    # For a new model, use:
    # model = YOLO("yolo26n.pt")

    model = YOLO(r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\runs\detect\train-60m640r\weights\best.pt")
    model.train(
        data=r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\data.yaml",
        epochs=40,           # Train longer, let patience handle the stopping
        # patience=20,
        imgsz=640,            # Start at 640; it's faster and more stable than 960
        batch=-1,             # Let AutoBatch optimize memory
        workers=4,            # Feed the GPU faster
        cache='disk'
        # degrees=15.0,
        # scale=0.5,
        # dropout=0.1,
        # perspective=0.001,
        # mixup=0.0,
        # copy_paste=0.0
    )