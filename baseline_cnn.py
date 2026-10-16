import tensorflow as tf
from data_loader import train_dataset, val_dataset

# ==============================
# Configuration
# ==============================

IMG_SIZE = (224, 224)
NUM_CLASSES = 1

# ==============================
# Build Baseline CNN
# ==============================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(224, 224, 3)),

    # Convolution Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Dropout
    tf.keras.layers.Dropout(0.5),

    # Binary Output
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# ==============================
# Display Model
# ==============================

model.summary()

# ==============================
# Training
# ==============================

print("\nStarting baseline CNN training...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# ==============================
# Save Model
# ==============================

model.save("models/baseline_cnn.keras")

print("\nBaseline CNN training completed!")
print("Model saved to: models/baseline_cnn.keras")