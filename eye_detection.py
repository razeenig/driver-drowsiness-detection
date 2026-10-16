import cv2
import mediapipe as mp
import math
import time
from playsound3 import playsound
import threading


# =========================
# SETTINGS
# =========================
EAR_THRESHOLD = 0.21
CLOSED_TIME = 2.0
AWAKE_TIME = 3.0

ALARM_FILE = "alarm.mp3"


# =========================
# FUNCTIONS
# =========================
def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_ear(landmarks, eye):
    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    return (vertical1 + vertical2) / (2.0 * horizontal)


# =========================
# EYE LANDMARKS
# =========================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Camera started!")
print("Driver Drowsiness Detection is running.")
print("Keep your eyes closed for 2 seconds to test the alarm.")
print("Press Q to quit.")


# =========================
# MEDIAPIPE
# =========================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)


# =========================
# VARIABLES
# =========================
closed_start = None
awake_start = None

alarm_active = False
alarm_thread = None


# =========================
# ALARM FUNCTION
# =========================
def play_alarm():
    global alarm_active

    while alarm_active:
        try:
            playsound(ALARM_FILE)
        except Exception as e:
            print("Alarm error:", e)
            break


# =========================
# MAIN LOOP
# =========================
with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        status = "NO FACE"
        ear = 0.0

        # =========================
        # FACE DETECTED
        # =========================
        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)

            ear = (left_ear + right_ear) / 2

            # =========================
            # EYES CLOSED
            # =========================
            if ear < EAR_THRESHOLD:

                awake_start = None

                if closed_start is None:
                    closed_start = time.time()

                closed_duration = time.time() - closed_start

                if closed_duration >= CLOSED_TIME:

                    status = "DROWSY"

                    # Start alarm only once
                    if not alarm_active:

                        alarm_active = True

                        alarm_thread = threading.Thread(
                            target=play_alarm,
                            daemon=True
                        )

                        alarm_thread.start()

                        print("DROWSINESS DETECTED!")
                        print("ALARM STARTED!")

                else:
                    status = "EYES CLOSING"

            # =========================
            # EYES OPEN
            # =========================
            else:

                closed_start = None

                # If alarm is active, require
                # 3 continuous seconds awake
                if alarm_active:

                    if awake_start is None:
                        awake_start = time.time()

                    awake_duration = time.time() - awake_start

                    status = "WAKE UP - ALARM ACTIVE"

                    # Stop alarm after 3 seconds awake
                    if awake_duration >= AWAKE_TIME:

                        alarm_active = False
                        awake_start = None

                        print("Person awake for 3 seconds.")
                        print("ALARM STOPPED!")

                        status = "AWAKE"

                else:

                    awake_start = None
                    status = "AWAKE"

        # =========================
        # NO FACE
        # =========================
        else:

            closed_start = None
            awake_start = None

            status = "NO FACE"

        # =========================
        # DISPLAY
        # =========================
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if alarm_active else (0, 255, 0),
            2
        )

        # Show closed-eye timer
        if closed_start is not None:

            closed_duration = time.time() - closed_start

            cv2.putText(
                frame,
                f"Closed: {closed_duration:.1f}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        # Show awake timer
        if awake_start is not None:

            awake_duration = time.time() - awake_start

            cv2.putText(
                frame,
                f"Awake: {awake_duration:.1f}s / 3.0s",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# =========================
# CLEANUP
# =========================
alarm_active = False

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")