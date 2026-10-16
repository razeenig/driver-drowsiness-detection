import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load training data
df = pd.read_csv("results/train.csv")

# Select 8 random images
samples = df.sample(8, random_state=42)

# Create figure
plt.figure(figsize=(12, 8))

for i, (_, row) in enumerate(samples.iterrows()):

    image = Image.open(row["image_path"])

    plt.subplot(2, 4, i + 1)
    plt.imshow(image)

    label = "Drowsy" if row["label"] == 1 else "Non Drowsy"

    plt.title(label)
    plt.axis("off")

plt.tight_layout()

# Save result
plt.savefig("results/dataset_samples.png")

plt.show()

print("Dataset visualization saved successfully!")
print("Location: results/dataset_samples.png")