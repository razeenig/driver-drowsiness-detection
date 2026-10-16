import os
from PIL import Image

dataset_path = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"

classes = ["Drowsy", "Non Drowsy"]

total_images = 0

print("=== Driver Drowsiness Dataset Inspection ===\n")

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{class_name}: {len(images)} images")

    total_images += len(images)

    if images:
        first_image = os.path.join(class_path, images[0])

        try:
            with Image.open(first_image) as img:
                print(f"  Example image: {images[0]}")
                print(f"  Image size: {img.size}")
                print(f"  Image mode: {img.mode}")
        except Exception as e:
            print(f"  Error reading image: {e}")

    print()

print(f"Total images: {total_images}")