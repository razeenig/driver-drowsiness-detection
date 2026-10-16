import os
import re
from collections import Counter

DATASET_DIR = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"

for class_name in ["Drowsy", "Non Drowsy"]:

    class_folder = os.path.join(DATASET_DIR, class_name)

    filenames = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print("\n================================")
    print(class_name)
    print("================================")

    print(f"Total images: {len(filenames)}")

    # Show first 50 filenames
    print("\nFirst 50 filenames:")

    for filename in sorted(filenames)[:50]:
        print(filename)

    # Analyze filename prefixes
    prefixes = Counter()

    for filename in filenames:

        name = os.path.splitext(filename)[0]

        match = re.match(r"([A-Za-z]+)", name)

        if match:
            prefixes[match.group(1)] += 1

    print("\nFilename prefixes:")

    for prefix, count in prefixes.most_common():
        print(f"{prefix}: {count}")