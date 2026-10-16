import os
import pandas as pd
import imagehash
from PIL import Image
from collections import defaultdict

BASE = r"C:\Users\razee\OneDrive\Desktop\DDD"

TRAIN_CSV = os.path.join(BASE, "results", "train.csv")
VAL_CSV = os.path.join(BASE, "results", "validation.csv")
TEST_CSV = os.path.join(BASE, "results", "test.csv")

THRESHOLD = 4

# --------------------------------------------------
# Load split files
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Dataset split:")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# --------------------------------------------------
# Function to calculate pHash
# --------------------------------------------------

def calculate_hashes(df, split_name):

    hashes = {}

    print(f"\nHashing {split_name}...")

    total = len(df)

    for i, path in enumerate(df["image_path"], start=1):

        try:
            img = Image.open(path).convert("RGB")
            hashes[path] = imagehash.phash(img)

        except Exception as e:
            print(f"Error: {path}")
            print(e)

        if i % 1000 == 0:
            print(f"  {i}/{total}")

    print(f"Finished {split_name}: {len(hashes)} images")

    return hashes


# --------------------------------------------------
# Calculate hashes
# --------------------------------------------------

train_hashes = calculate_hashes(train_df, "TRAIN")
val_hashes = calculate_hashes(val_df, "VALIDATION")
test_hashes = calculate_hashes(test_df, "TEST")


# --------------------------------------------------
# Create buckets from training hashes
#
# Instead of comparing every image with every image,
# we first put hashes into buckets.
# --------------------------------------------------

def create_buckets(hashes):

    buckets = defaultdict(list)

    for path, h in hashes.items():

        # Split the 64-bit hash into 4 sections
        hash_string = str(h)

        for i in range(4):

            bucket_key = (
                i,
                hash_string[i * 16:(i + 1) * 16]
            )

            buckets[bucket_key].append(path)

    return buckets


train_buckets = create_buckets(train_hashes)

print("\nTraining hash buckets created.")
print(f"Number of buckets: {len(train_buckets)}")


# --------------------------------------------------
# Find similar images
# --------------------------------------------------

def check_leakage(target_hashes, target_name):

    print("\n========================================")
    print(f"TRAIN vs {target_name.upper()}")
    print("========================================")

    similar_images = 0
    total = len(target_hashes)

    examples = []

    for index, (target_path, target_hash) in enumerate(
        target_hashes.items(), start=1
    ):

        candidates = set()

        hash_string = str(target_hash)

        # Retrieve only possible candidates
        for i in range(4):

            bucket_key = (
                i,
                hash_string[i * 16:(i + 1) * 16]
            )

            candidates.update(train_buckets.get(bucket_key, []))

        found = False

        for train_path in candidates:

            distance = target_hash - train_hashes[train_path]

            if distance <= THRESHOLD:

                similar_images += 1
                found = True

                if len(examples) < 10:
                    examples.append(
                        (
                            target_path,
                            train_path,
                            distance
                        )
                    )

                break

        if index % 500 == 0:
            print(
                f"Checked {index}/{total} | "
                f"similar: {similar_images}"
            )

    percentage = (
        similar_images / total * 100
        if total > 0 else 0
    )

    print("\nRESULT")
    print("----------------------------------------")
    print(f"Target split: {target_name}")
    print(f"Total images: {total}")
    print(f"Similar to training: {similar_images}")
    print(f"Percentage: {percentage:.2f}%")

    print("\nExamples:")

    for target, train, distance in examples:

        print(f"\nTarget: {target}")
        print(f"Train:  {train}")
        print(f"pHash distance: {distance}")


# --------------------------------------------------
# Check validation and test
# --------------------------------------------------

check_leakage(
    val_hashes,
    "validation"
)

check_leakage(
    test_hashes,
    "test"
)