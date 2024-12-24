import cv2
import mediapipe as mp
import time
import pyautogui

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands and Drawing utilities
mphands = mp.solutions.hands
hands = mphands.Hands(False)
mpdraw = mp.solutions.drawing_utils

pTime = 0  # Previous Time
cTime = 0  # Current Time

# Get screen width and height for mouse mapping
screen_width, screen_height = pyautogui.size()

while True:
    success, img = cap.read()  # Capture frame
    if not success:
        print("Failed to capture image from webcam")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = hands.process(imgRGB)  # Process the image for hand landmarks

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Initialize variables to store fingertip positions
            thumb_tip = handLms.landmark[4]
            index_tip = handLms.landmark[8]
            middle_tip = handLms.landmark[12]
            ring_tip = handLms.landmark[16]
            pinky_tip = handLms.landmark[18]

            # Loop through hand landmarks for mouse movement
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Get pixel coordinates
                if id == 8:  # Track the tip of the index finger (ID 8)
                    # Map hand coordinates to screen coordinates
                    x = screen_width - cx
                    y = cy
                    pyautogui.moveTo(x, y)  # Move the mouse

            # Calculate the distance between the thumb and index finger to detect a pinch
            # Calculate the distance between the middle finger and index finger to detect scrolling
            # Calculate the distance between the pinky finger and ring finger to detect right click

            distanceLC = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            distanceScroll = ((middle_tip.x - index_tip.x) ** 2 + (middle_tip.y - index_tip.y) ** 2) ** 0.5
            distanceRC = ((ring_tip.x - pinky_tip.x) ** 2 + (ring_tip.y - pinky_tip.y) ** 2) ** 0.5

            # Click detection if fingers are close enough
            if distanceLC < 0.05:  # If fingers are close enough, consider it a click
                pyautogui.click()

            if distanceScroll > 0.05:
                pyautogui.rightClick()

            # Scroll logic using the distance between middle and index finger
            if distanceScroll < 0.05:  # Fingers are not too close
                # Compare the Y positions to determine scroll direction
                if middle_tip.y < index_tip.y:  # Middle finger is above index finger
                    pyautogui.scroll(40)  # Scroll up
                elif middle_tip.y > index_tip.y:  # Middle finger is below index finger
                    pyautogui.scroll(-40)  # Scroll down



            # Draw hand landmarks and connections
            mpdraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0  # Avoid ZeroDivisionError
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 3)

    # Show the image
    cv2.imshow("Image", img)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
