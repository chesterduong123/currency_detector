from ultralytics import YOLO

model = YOLO(r"C:\Users\chest\OneDrive\Documents\YOLOProjects\runs\detect\train-24\weights\best.pt")

results = model.predict(
    source=[r"C:\Users\chest\OneDrive\Pictures\Camera Roll 1\WIN_20260614_15_51_00_Pro.jpg",
            r"C:\Users\chest\OneDrive\Pictures\Camera Roll 1\WIN_20260614_16_51_51_Pro.jpg",
            "https://thumbs.dreamstime.com/b/coins-table-13836783.jpg?w=992",
            "https://i.ebayimg.com/images/g/eT0AAOSw5Vhmfd4n/s-l1200.jpg",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpFoGXyjzDeRW-t7UfZdVj1QXQNyLvR9obNg&s",
            "https://i.gyazo.com/2ee0d44a92e288cac3b897eadf30eccc.png",
            "https://preview.redd.it/all-us-dollar-coins-and-bills-v0-hkhds0avyr9b1.jpg?width=640&crop=smart&auto=webp&s=18f45538186747384232654eadd30d5f5c2e6074",
            r"C:\Users\chest\OneDrive\Documents\YOLOProjects\currency\test\images\IMG_2092_jpg.rf.b6fa33cc1d9459dfd24f6e68e03d9bb8.jpg"],
    save=True,
    show=False,
    conf=0.5
)

# test fifth pull

print("Prediction complete!")
