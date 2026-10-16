import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================
# Configuration
# ==============================

DATASET_DIR = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"
OUTPUT_DIR = r"C:\Users\razee\OneDrive\Desktop\DDD\results"

RANDOM_STATE = 42

# ==============================
# Load image paths and labels
# ==============================

image_paths = []
labels = []

classes = {
    "Drowsy": 1,
    "Non Drowsy": 0
}

for class_name, label in classes.items():

    class_folder = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(class_folder):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(class_folder, filename)

            image_paths.append(image_path)
            labels.append(label)

print(f"Total images found: {len(image_paths)}")

# ==============================
# Create DataFrame
# ==============================

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

print("\nClass distribution:")
print(df["label"].value_counts())

# ==============================
# Train / Validation / Test Split
# ==============================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)

# ==============================
# Save split files
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

# ==============================
# Results
# ==============================

print("\nDataset split completed!")

print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Testing images:    {len(test_df)}")

print("\nCSV files saved in:")
print(OUTPUT_DIR)