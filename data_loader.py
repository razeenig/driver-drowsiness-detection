import pandas as pd
import tensorflow as tf

# ==============================
# Configuration
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# Load CSV files
# ==============================

train_df = pd.read_csv("results/train.csv")
val_df = pd.read_csv("results/validation.csv")
test_df = pd.read_csv("results/test.csv")


# ==============================
# Image loading function
# ==============================

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ==============================
# Create TensorFlow datasets
# ==============================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df["image_path"].values, train_df["label"].values)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df["image_path"].values, val_df["label"].values)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_df["image_path"].values, test_df["label"].values)
)


# ==============================
# Data augmentation
# ==============================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10)
])


# ==============================
# Training preprocessing
# ==============================

def prepare_train(image_path, label):

    image, label = load_image(image_path, label)

    image = data_augmentation(image)

    return image, label


# ==============================
# Validation/Test preprocessing
# ==============================

def prepare_eval(image_path, label):

    return load_image(image_path, label)


# ==============================
# Build pipelines
# ==============================

train_dataset = (
    train_dataset
    .shuffle(5000)
    .map(prepare_train, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(prepare_eval, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
    .map(prepare_eval, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ==============================
# Display information
# ==============================

print("Data loader created successfully!")

print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Testing images:    {len(test_df)}")

print(f"Image size:        {IMG_SIZE}")
print(f"Batch size:        {BATCH_SIZE}")