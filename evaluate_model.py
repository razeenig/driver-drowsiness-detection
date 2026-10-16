import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import test_dataset

# ==============================
# Load trained model
# ==============================

MODEL_PATH = "models/baseline_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ==============================
# Evaluate on test dataset
# ==============================

print("\nEvaluating on test dataset...\n")

results = model.evaluate(test_dataset, verbose=1)

print("\nTest Results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ==============================
# Generate predictions
# ==============================

predictions = model.predict(test_dataset)

predicted_labels = (predictions >= 0.5).astype(int).flatten()

# Get actual labels
test_df = pd.read_csv("results/test.csv")
actual_labels = test_df["label"].values

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(actual_labels, predicted_labels)

print("\nConfusion Matrix:")
print(cm)

# ==============================
# Classification Report
# ==============================

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=["Non Drowsy", "Drowsy"],
        digits=4
    )
)