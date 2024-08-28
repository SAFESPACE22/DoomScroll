import mediapipe as mp
import cv2
import time
from autoScroll import scroll_function

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

prev_finger_y = 0
scroll_triggered = False  # Flag to track if scroll action has been triggered
scroll_threshold = 100  # Distance threshold for scrolling (adjust as needed)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 8:  # Index finger landmark
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                    
                    # Check if the index finger has moved a certain distance and the scroll action hasn't been triggered yet
                    if prev_finger_y != 0 and (prev_finger_y - cy) > scroll_threshold and not scroll_triggered:
                        scroll_function()  # Call your scroll function here
                        scroll_triggered = True  # Set flag to true to prevent repeated scrolling
                    
                    # Reset scroll action flag if finger is no longer moving up
                    if (prev_finger_y - cy) <= 0:
                        scroll_triggered = False
                    
                    prev_finger_y = cy
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
