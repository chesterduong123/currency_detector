from ultralytics import YOLO

model = YOLO(r"C:\Users\chest\OneDrive\Documents\YOLO26 Stuff\Currency Project\runs\detect\train-100m640r\weights\best.pt")

results = model.predict(
    source=[r"C:\Users\chest\OneDrive\Pictures\Camera Roll 1\WIN_20260614_15_51_00_Pro.jpg",
            r"C:\Users\chest\OneDrive\Pictures\Camera Roll 1\WIN_20260614_16_51_51_Pro.jpg",
            "https://thumbs.dreamstime.com/b/coins-table-13836783.jpg?w=992",
            "https://i.ebayimg.com/images/g/eT0AAOSw5Vhmfd4n/s-l1200.jpg",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpFoGXyjzDeRW-t7UfZdVj1QXQNyLvR9obNg&s",
            "https://i.gyazo.com/2ee0d44a92e288cac3b897eadf30eccc.png",
            "https://preview.redd.it/all-us-dollar-coins-and-bills-v0-hkhds0avyr9b1.jpg?width=640&crop=smart&auto=webp&s=18f45538186747384232654eadd30d5f5c2e6074",
            "https://i.ebayimg.com/images/g/~HkAAOSw-S5nhwD~/s-l1200.jpg",
            "https://www.wikihow.com/images/thumb/d/da/Count-Money-Step-2-Version-4.jpg/v4-460px-Count-Money-Step-2-Version-4.jpg",
            "https://images.ctfassets.net/h9i30ml9c3lu/15GlTLd6ibuOs1KNomsThi/952882051f6ee5a5beed174bc0c7de75/kenny-eliason-maJDOJSmMoo-unsplash__1_.jpg",
            "https://preview.redd.it/cant-wait-for-the-new-2026-10-bill-redesign-v0-udmuasaytb4g1.jpeg?width=1080&crop=smart&auto=webp&s=b27559131368182991494736c2a4d4d66c37ea59",
            "https://preview.redd.it/what-does-the-new-us-currency-look-like-v0-zeo7vvtgeqo81.jpg?width=1311&format=pjpg&auto=webp&s=b92f4a84a7179c8796d569755f492774d65802bb",
            "https://preview.redd.it/are-there-any-changes-that-you-would-make-to-the-face-on-v0-nfxvhp97icib1.jpg?width=1080&crop=smart&auto=webp&s=573170d89abc34c18fe6e2bf35f6dc26aea983f5"],
    save=True,
    show=False,
    conf=0.25,
    imgsz=640
)

print("Prediction complete!")
