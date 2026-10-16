import cv2
import mediapipe as mp

STREAM_URL = "http://192.168.0.198:8080/stream.mjpg"

# MediaPipe
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

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Could not connect to iPhone stream.")
    exit()

print("Face detection started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect faces
    result = detector.detect(mp_image)

    # Draw boxes
    for detection in result.detections:

        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()