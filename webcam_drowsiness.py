import cv2
import numpy as np
import tensorflow as tf

# =========================
# SETTINGS
# =========================
MODEL_PATH = "models/baseline_cnn.keras"
IMG_SIZE = 224

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixels
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0][0]

    # Our labels:
    # 0 = Non Drowsy
    # 1 = Drowsy
    if prediction >= 0.5:
        label = "DROWSY"
        confidence = prediction * 100
    else:
        label = "NON-DROWSY"
        confidence = (1 - prediction) * 100

    # Display result
    text = f"{label} - {confidence:.1f}%"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if label == "DROWSY" else (0, 255, 0),
        2
    )

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()