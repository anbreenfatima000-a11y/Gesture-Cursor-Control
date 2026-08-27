import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def is_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    folded_count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_count += 1
    return folded_count >= 3

def is_two_fingers_up(hand_landmarks):
    index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_extended = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_folded = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_folded = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    return index_extended and middle_extended and ring_folded and pinky_folded

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0
smoothening = 3
frameR = 100

# Click / Double-click / Drag state
fist_start_time = None
is_holding_drag = False
hold_threshold = 1       
last_click_time = 0
double_click_threshold = 0.3
scroll_confirm_count = 0
scroll_confirm_threshold = 4  

# Scroll state
prev_scroll_y = None
scroll_sensitivity = 3

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    cv2.rectangle(frame, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            palm_center = hand_landmarks.landmark[9]
            fist_detected = is_fist(hand_landmarks)
            two_fingers_detected = is_two_fingers_up(hand_landmarks)

            x_pixel = palm_center.x * w
            y_pixel = palm_center.y * h

            if two_fingers_detected:
                scroll_confirm_count += 1
            else:
                scroll_confirm_count = 0

            if scroll_confirm_count >= scroll_confirm_threshold:
                # Safety: agar drag active thi, usay pehle close karein
                if is_holding_drag:
                    pyautogui.mouseUp()
                    is_holding_drag = False
                    print("Drag End (interrupted by scroll)")
                fist_start_time = None

                # ---- SCROLL MODE ----
                if prev_scroll_y is not None:
                    delta = prev_scroll_y - y_pixel
                    if abs(delta) > 2:
                        pyautogui.scroll(int(delta * scroll_sensitivity))
                prev_scroll_y = y_pixel

                cv2.putText(frame, "SCROLL MODE", (50, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            else:
                prev_scroll_y = None

                # ---- CURSOR MOVEMENT (baaki sab pehle jaisa) ----
                screen_x = np.interp(x_pixel, [frameR, w - frameR], [0, screen_w])
                screen_y = np.interp(y_pixel, [frameR, h - frameR], [0, screen_h])
                screen_x = np.clip(screen_x, 0, screen_w)
                screen_y = np.clip(screen_y, 0, screen_h)

                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                if fist_detected:
                    if fist_start_time is None:
                        fist_start_time = time.time()

                    hold_duration = time.time() - fist_start_time

                    if hold_duration >= hold_threshold and not is_holding_drag:
                        pyautogui.mouseDown()
                        is_holding_drag = True
                        print("Drag Start")

                    cv2.putText(frame, "FIST - HOLD/CLICK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    if is_holding_drag:
                        pyautogui.mouseUp()
                        is_holding_drag = False
                        print("Drag End")

                    elif fist_start_time is not None:
                        current_time = time.time()
                        time_since_last_click = current_time - last_click_time

                        if time_since_last_click < double_click_threshold:
                            pyautogui.doubleClick()
                            print("Double Click!")
                        else:
                            pyautogui.click()
                            print("Single Click")

                        last_click_time = current_time

                    fist_start_time = None

                # ---- CURSOR MOVEMENT ----
                screen_x = np.interp(x_pixel, [frameR, w - frameR], [0, screen_w])
                screen_y = np.interp(y_pixel, [frameR, h - frameR], [0, screen_h])
                screen_x = np.clip(screen_x, 0, screen_w)
                screen_y = np.clip(screen_y, 0, screen_h)

                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # ---- CLICK / DOUBLE-CLICK / DRAG LOGIC ----
                if fist_detected:
                    if fist_start_time is None:
                        fist_start_time = time.time()

                    hold_duration = time.time() - fist_start_time

                    if hold_duration >= hold_threshold and not is_holding_drag:
                        pyautogui.mouseDown()
                        is_holding_drag = True
                        print("Drag Start")

                    cv2.putText(frame, "FIST - HOLD/CLICK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    if is_holding_drag:
                        pyautogui.mouseUp()
                        is_holding_drag = False
                        print("Drag End")

                    elif fist_start_time is not None:
                        
                        current_time = time.time()
                        time_since_last_click = current_time - last_click_time

                        if time_since_last_click < double_click_threshold:
                            pyautogui.doubleClick()
                            print("Double Click!")
                        else:
                            pyautogui.click()
                            print("Single Click")

                        last_click_time = current_time

                    fist_start_time = None

            cv2.circle(frame, (int(x_pixel), int(y_pixel)), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Gesture Cursor Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()