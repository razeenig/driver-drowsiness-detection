import os
from collections import defaultdict

import imagehash
from PIL import Image

DATASET_DIR = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"

# Images with pHash distance <= this value will be considered very similar.
THRESHOLD = 4

hashes = []

print("Calculating perceptual hashes...")
print("This may take a few minutes.\n")

for class_name in ["Drowsy", "Non Drowsy"]:

    class_folder = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_folder):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(class_folder, filename)

            try:
                image = Image.open(image_path).convert("RGB")
                image_hash = imagehash.phash(image)

                hashes.append(
                    (image_hash, image_path, class_name)
                )

            except Exception as e:
                print(f"Could not read: {image_path}")
                print(e)

print(f"\nImages processed: {len(hashes)}")

# ==============================
# Find similar images
# ==============================

similar_pairs = []

print("\nSearching for highly similar images...")

for i in range(len(hashes)):

    hash1, path1, class1 = hashes[i]

    for j in range(i + 1, len(hashes)):

        hash2, path2, class2 = hashes[j]

        distance = hash1 - hash2

        if distance <= THRESHOLD:

            similar_pairs.append(
                (distance, path1, path2, class1, class2)
            )

# ==============================
# Results
# ==============================

print("\n================================")
print("Near-Duplicate Analysis")
print("================================")

print(f"Similarity threshold: {THRESHOLD}")
print(f"Similar image pairs: {len(similar_pairs)}")

if similar_pairs:

    print("\nExamples:")

    for distance, path1, path2, class1, class2 in similar_pairs[:20]:

        print("\n------------------------------")
        print(f"Hash distance: {distance}")
        print(f"Image 1: {path1}")
        print(f"Class 1:  {class1}")
        print(f"Image 2: {path2}")
        print(f"Class 2:  {class2}")

else:

    print("\nNo highly similar image pairs found.")