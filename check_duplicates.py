import os
import hashlib
from collections import defaultdict

DATASET_DIR = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"

hashes = defaultdict(list)

print("Scanning dataset for exact duplicate images...\n")

for class_name in ["Drowsy", "Non Drowsy"]:

    class_folder = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_folder):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(class_folder, filename)

            with open(image_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            hashes[file_hash].append(image_path)

# Find duplicate groups
duplicate_groups = [
    paths for paths in hashes.values()
    if len(paths) > 1
]

total_duplicate_files = sum(
    len(paths) for paths in duplicate_groups
)

print(f"Total unique image files: {len(hashes)}")
print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Files involved in duplicates: {total_duplicate_files}")

if duplicate_groups:

    print("\nExample duplicate groups:")

    for group in duplicate_groups[:10]:

        print("\n--- Duplicate group ---")

        for path in group:
            print(path)

else:

    print("\nNo exact duplicate images found.")