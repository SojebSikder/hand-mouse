import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open the webcam
cap = cv2.VideoCapture(0)
# Initialize previous cursor position for smoothing
prev_x, prev_y = 0, 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # Get frame dimensions

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract index finger tip (landmark 8) and thumb tip (landmark 4)
            index_finger = hand_landmarks.landmark[8]
            thumb_finger = hand_landmarks.landmark[4]

            # Convert normalized coordinates to screen coordinates
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)
            screen_x = np.interp(x, [0, w], [0, screen_width])
            screen_y = np.interp(y, [0, h], [0, screen_height])

            # Apply smoothing for cursor movement
            prev_x = (prev_x * 0.8) + (screen_x * 0.2)
            prev_y = (prev_y * 0.8) + (screen_y * 0.2)
            pyautogui.moveTo(prev_x, prev_y, duration=0.1)

            # Calculate Euclidean distance between index finger tip and thumb tip
            thumb_x = int(thumb_finger.x * w)
            thumb_y = int(thumb_finger.y * h)
            distance = np.hypot(thumb_x - x, thumb_y - y)

            # Click when fingers come close together
            if distance < 40:
                pyautogui.click()

            # Draw landmarks on hand
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
