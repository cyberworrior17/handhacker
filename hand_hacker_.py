import cv2
import mediapipe as mp
import time
import winsound

# ====== MediaPipe Hand Setup ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# ====== Camera ======
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

scan_start = None
progress = 0
unlocked = False

def beep(freq=1200, dur=120):
    try:
        winsound.Beep(freq, dur)
    except:
        pass

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Black background overlay (hacker style)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # ====== LOGO ======
    cv2.putText(frame, "MR.VIRUS SYSTEM",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, "HAND BIOMETRIC INTERFACE",
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 200, 0), 1)

    if results.multi_hand_landmarks:
        if scan_start is None:
            scan_start = time.time()
            progress = 0
            unlocked = False
            beep(1000, 150)

        elapsed = time.time() - scan_start
        progress = min(int((elapsed / 4) * 100), 100)

        # ====== Loading Text ======
        cv2.putText(frame, "SCANNING HAND...",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        # ====== Loading Bar ======
        bar_x, bar_y = 30, 150
        bar_w, bar_h = 400, 25

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (0, 255, 0), 2)

        fill_w = int((progress / 100) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h),
                      (0, 255, 0), -1)

        cv2.putText(frame, f"{progress}%",
                    (bar_x + bar_w + 10, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        if progress >= 100 and not unlocked:
            unlocked = True
            beep(1600, 300)
            time.sleep(0.3)

        if unlocked:
            cv2.putText(frame, "SYSTEM UNLOCKED",
                        (30, 220), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 0), 3)

            cv2.putText(frame, "WELCOME MR.VIRUS",
                        (30, 270), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.putText(frame, "INSTAGRAM: secured@uknown_virus404x",
                        (30, 310), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 0), 2)

    else:
        scan_start = None
        progress = 0
        unlocked = False

        cv2.putText(frame, "PLACE HAND IN FRONT OF CAMERA",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 180, 0), 2)

    cv2.imshow("HAND SCAN SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
