import cv2
try:
    import pydirectinput as pyautogui
except ImportError:
    import pyautogui
import time
import numpy as np

THRESHOLD_SENSITIVITY = 25
MIN_CONTOUR_AREA = 700

ZONE_WIDTH_RATIO = 0.25
ZONE_HEIGHT_RATIO = 0.40 # Altura da linha de pulo
MARGIN_X_RATIO = 0.05

JUMP_KEY = 'space'
JUMP_LINE_Y_RATIO = 0.4 

# --- TECLAS ---
LEFT_KEY = 'left'
RIGHT_KEY = 'right'

FRAMES_TO_ENTER_ZONE = 2
FRAMES_TO_LEAVE_ZONE = 4


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

keys_down = { 
    LEFT_KEY: False, 
    RIGHT_KEY: False,
    JUMP_KEY: False
}
motion_counters = { 
    LEFT_KEY: 0, 
    RIGHT_KEY: 0,
    JUMP_KEY: 0
}

prev_frame = None 

print("Controle Left/Right + Pulo iniciado. Pressione 'q' para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        

        zone_w = int(width * ZONE_WIDTH_RATIO)
        zone_h = int(height * ZONE_HEIGHT_RATIO)
        margin_x = int(width * MARGIN_X_RATIO)
        zone_y = (height // 2) - (zone_h // 2)

        left_zone_rect = (margin_x, zone_y, zone_w, zone_h)
        right_zone_x = width - zone_w - margin_x
        right_zone_rect = (right_zone_x, zone_y, zone_w, zone_h)
        jump_line_y = int(height * JUMP_LINE_Y_RATIO)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)
        if prev_frame is None:
            prev_frame = gray
            continue
        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_in_left_zone, motion_in_right_zone = False, False
        current_highest_motion_y = height
        found_significant_motion = False

        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA: continue
            
            found_significant_motion = True
            (x_cont, y_cont, _, _) = cv2.boundingRect(contour)
            current_highest_motion_y = min(current_highest_motion_y, y_cont)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                if left_zone_rect[0] < cx < left_zone_rect[0] + left_zone_rect[2] and \
                   left_zone_rect[1] < cy < left_zone_rect[1] + left_zone_rect[3]:
                    motion_in_left_zone = True
                elif right_zone_rect[0] < cx < right_zone_rect[0] + right_zone_rect[2] and \
                     right_zone_rect[1] < cy < right_zone_rect[1] + right_zone_rect[3]:
                    motion_in_right_zone = True
        
        motion_in_jump_zone = found_significant_motion and (current_highest_motion_y < jump_line_y)
        
        all_zones = [
            (LEFT_KEY, motion_in_left_zone),
            (RIGHT_KEY, motion_in_right_zone),
            (JUMP_KEY, motion_in_jump_zone) 
        ]

        for key, motion_detected in all_zones:
            if motion_detected:
                motion_counters[key] = min(FRAMES_TO_ENTER_ZONE, motion_counters[key] + 1)
            else:
                motion_counters[key] = max(-FRAMES_TO_LEAVE_ZONE, motion_counters[key] - 1)

            if motion_counters[key] == FRAMES_TO_ENTER_ZONE and not keys_down[key]:
                pyautogui.keyDown(key); keys_down[key] = True; print(f"KeyDown '{key}'")
            elif motion_counters[key] == -FRAMES_TO_LEAVE_ZONE and keys_down[key]:
                pyautogui.keyUp(key); keys_down[key] = False; print(f"KeyUp '{key}'")

        # --- RETORNO VISUAL DA WEBCAM ---
        left_color = (0, 255, 0) if keys_down[LEFT_KEY] else (255, 100, 100)
        right_color = (0, 255, 0) if keys_down[RIGHT_KEY] else (255, 100, 100)
        jump_line_color = (0, 255, 0) if keys_down[JUMP_KEY] else (255, 0, 0)
        
        cv2.rectangle(frame, left_zone_rect, left_color, 2)
        cv2.rectangle(frame, right_zone_rect, right_color, 2)
        cv2.line(frame, (0, jump_line_y), (width, jump_line_y), jump_line_color, 2)
        
        def draw_text(rect, text):
            cv2.putText(frame, text, (rect[0] + rect[2]//2 - 30, rect[1] + rect[3]//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
        draw_text(left_zone_rect, LEFT_KEY.upper())
        draw_text(right_zone_rect, RIGHT_KEY.upper())
        
        cv2.imshow("Controle Left/Right + Pulo - Pressione 'q' para sair", frame)
        
        prev_frame = gray
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    print("\nFechando...")
    for key, is_down in keys_down.items():
        if is_down: pyautogui.keyUp(key)
    cap.release()
    cv2.destroyAllWindows()