import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# =========================
# SETTINGS
# =========================

STREAM_URL = "http://192.168.0.198:8080/stream.mjpg"

MODEL_PATH = "models/baseline_cnn.keras"

IMG_SIZE = 224

# =========================
# LOAD CNN
# =========================

print("Loading drowsiness model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =========================
# MEDIAPIPE FACE DETECTOR
# =========================

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="models/blaze_face_short_range.tflite"
    ),
    running_mode=VisionRunningMode.IMAGE
)

detector = FaceDetector.create_from_options(options)

# =========================
# OPEN IPHONE CAMERA
# =========================

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("ERROR: Could not open iPhone camera.")
    exit()

print("iPhone camera connected!")
print("Press Q to quit.")

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect face
    result = detector.detect(mp_image)

    for detection in result.detections:

        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height

        # Keep coordinates inside image
        x = max(0, x)
        y = max(0, y)

        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        # Crop face
        face = frame[y:y2, x:x2]

        if face.size == 0:
            continue

        # Resize face
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        # BGR → RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize
        face = face.astype(np.float32) / 255.0

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # CNN prediction
        prediction = model.predict(face, verbose=0)[0][0]

        # Our dataset:
        # 0 = Non Drowsy
        # 1 = Drowsy

        if prediction >= 0.5:
            label = "DROWSY"
            confidence = prediction
        else:
            label = "NON-DROWSY"
            confidence = 1 - prediction

        # Draw face box
        cv2.rectangle(
            frame,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display result
        text = f"{label} ({confidence * 100:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show camera
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()