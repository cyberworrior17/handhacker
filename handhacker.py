import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import random
import math

# ================== SOUND ==================
pygame.mixer.init()
typing_sound = pygame.mixer.Sound("typing.wav")     # typing loop
accept_sound = pygame.mixer.Sound("accept.wav")    # beep ya acceptance

# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7
)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

WINDOW_NAME = "MRVIRUS SYSTEM"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
is_fullscreen = False

# ================== STATE VARS ==================
scan_y = 0
scan_dir = 1
loading = 0
hand_detected = False
start_time = None

# typing control
text_start_time = None
typed_chars = 0
show_accept = False
accept_played = False

# screen shake
shake_frames = 0

# ================== MATRIX ==================
h, w = 480, 640
cols = np.random.randint(0, h, size=w)

def draw_matrix(frame):
    global cols
    for i in range(0, w, 10):
        y = cols[i]
        cv2.putText(frame, chr(random.randint(33, 126)),
                    (i, y),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 1)
        cols[i] = y + random.randint(5, 15)
        if cols[i] > h:
            cols[i] = 0

# ================== TYPE EFFECT + CURSOR ==================
def type_text(frame, text, pos, speed=0.05):
    global typed_chars
    elapsed = time.time() - text_start_time
    chars = int(elapsed / speed)
    typed_chars = min(chars, len(text))

    # blinking cursor
    cursor_on = int(time.time() * 2) % 2 == 0
    cursor = "|" if cursor_on and typed_chars < len(text) else ""

    cv2.putText(frame, text[:typed_chars] + cursor,
                pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)
    return typed_chars == len(text)

# ================== SCREEN SHAKE ==================
def apply_shake(frame, intensity=8):
    dx = random.randint(-intensity, intensity)
    dy = random.randint(-intensity, intensity)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    draw_matrix(frame)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        if not hand_detected:
            start_time = time.time()
            typing_sound.play(-1)
            text_start_time = None
            typed_chars = 0
            show_accept = False
            accept_played = False
            shake_frames = 0

        hand_detected = True

        # ===== SENSOR LINE =====
        scan_y += scan_dir * 10
        if scan_y >= h or scan_y <= 0:
            scan_dir *= -1

        cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 0), 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, scan_y-10), (w, scan_y+10), (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        # ===== LOADING =====
        elapsed = time.time() - start_time
        loading = min(int(elapsed * 20), 100)

        cv2.rectangle(frame, (120, 400), (520, 420), (0, 255, 0), 2)
        cv2.rectangle(frame, (120, 400),
                      (120 + int(4 * loading), 420),
                      (0, 255, 0), -1)

        cv2.putText(frame, f"SCANNING... {loading}%",
                    (200, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # ===== RESULT =====
        if loading >= 100:
            typing_sound.stop()

            if text_start_time is None:
                text_start_time = time.time() + 1  # ⏱️ delay 1 sec

            if time.time() >= text_start_time:
                done = type_text(
                    frame,
                    "MRVIRUS HAND DETECTED",
                    (140, 160),
                    speed=0.05
                )
                if done:
                    show_accept = True

            if show_accept:
                if not accept_played:
                    accept_sound.play()
                    accept_played = True
                    shake_frames = 12  # number of shake frames

                cv2.putText(frame, "AND ACCEPTED",
                            (230, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

                cv2.putText(frame, "SYSTEM UNLOCKED",
                            (180, 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 3)

                cv2.putText(frame, "WELCOME MR.VIRUS",
                            (170, 285),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

                cv2.putText(frame,
                            "INSTAGRAM: secured@uknown_virus404x",
                            (80, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    else:
        hand_detected = False
        loading = 0
        typing_sound.stop()
        text_start_time = None
        typed_chars = 0
        show_accept = False
        accept_played = False
        shake_frames = 0

    # ===== APPLY SCREEN SHAKE =====
    if shake_frames > 0:
        frame = apply_shake(frame, intensity=8)
        shake_frames -= 1

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    # FULLSCREEN TOGGLE
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        mode = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

    # EXIT
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

