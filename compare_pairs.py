import os
from PIL import Image, ImageChops
import imagehash

BASE = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"

pairs = [
    ("A0002.png", "a0002.png"),
    ("A0003.png", "a0003.png"),
    ("A0010.png", "a0010.png"),
    ("X0100.png", "x0100.png"),
    ("ZB0100.png", "zb0100.png"),
]

for drowsy_file, non_file in pairs:

    drowsy_path = os.path.join(BASE, "Drowsy", drowsy_file)
    non_path = os.path.join(BASE, "Non Drowsy", non_file)

    if not os.path.exists(drowsy_path) or not os.path.exists(non_path):
        print(f"Missing pair: {drowsy_file} / {non_file}")
        continue

    img1 = Image.open(drowsy_path).convert("RGB")
    img2 = Image.open(non_path).convert("RGB")

    # Perceptual hashes
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    distance = hash1 - hash2

    # Average pixel difference
    diff = ImageChops.difference(img1, img2)

    pixels = list(diff.getdata())

    avg_difference = sum(
        sum(pixel) / 3 for pixel in pixels
    ) / len(pixels)

    print("\n-----------------------------")
    print(f"Drowsy:     {drowsy_file}")
    print(f"Non-Drowsy: {non_file}")
    print(f"pHash distance: {distance}")
    print(f"Average pixel difference: {avg_difference:.2f}")