import cv2
import mediapipe as mp
from deepface import DeepFace

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = face_mesh.process(imgRGB)

    # Draw landmarks on the face
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # Use DeepFace to analyze emotions
    result = DeepFace.analyze(img, actions=['emotion'])
    emotion = result[0]['dominant_emotion']
    cv2.putText(img, f'Emotion: {emotion}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    # Show the image with emotion label
    cv2.imshow("Emotion Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
