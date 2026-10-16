import cv2

STREAM_URL = "http://192.168.0.198:8080/stream.mjpg"

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Could not open iPhone stream.")
    exit()

print("iPhone stream connected!")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    cv2.imshow("iPhone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()