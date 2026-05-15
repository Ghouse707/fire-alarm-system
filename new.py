from ultralytics import YOLO
import cv2
import pygame
import sys

# -----------------------------
# Settings
# -----------------------------
CAMERA_INDEX = 0              # webcam
MODEL_PATH = "smoke.pt"
ALARM_PATH = "police.mp3"

DETECT_EVERY = 2
IMGSZ = 416
CONF_THRESHOLD = 0.25

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)
model.fuse()

# -----------------------------
# Load alarm sound
# -----------------------------
pygame.mixer.init()
pygame.mixer.music.load(ALARM_PATH)

# -----------------------------
# Open camera
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit()

frame_count = 0
fire_detected = False
last_detection_fire = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from camera")
        break

    frame_count += 1
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO only on selected frames
    if frame_count % DETECT_EVERY == 0:
        results = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        last_detection_fire = False

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls].lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if ("fire" in label or "smoke" in label) and conf >= CONF_THRESHOLD:
                    last_detection_fire = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"{label.upper()} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

    fire_detected = last_detection_fire if frame_count % DETECT_EVERY == 0 else fire_detected

    if fire_detected:
        cv2.putText(
            frame,
            "FIRE DETECTED!",
            (150, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)
    else:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    cv2.imshow("Fire Detection Alarm System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()