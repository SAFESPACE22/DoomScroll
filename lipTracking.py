import cv2
import mediapipe as mp
from autoScroll import scroll_function  # Ensure autoScroll module and function are correctly imported

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)  # Capture video from webcam (index 0)

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Parameters for lip detection
lip_open_threshold = 0.05  # Adjust this threshold to increase the sensitivity
lip_distance_history = []

with mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Use video stream mode (not static images)
    min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():  # Loop to continuously process frames
        ret, frame = cap.read()  # Read a frame from the webcam

        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        # Convert the BGR image to RGB before processing.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Print and draw face mesh landmarks on the image.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract specific landmarks for lips
                upper_lip_landmarks = face_landmarks.landmark[61:65]
                lower_lip_landmarks = face_landmarks.landmark[65:69]

                # Calculate the distance between upper and lower lip landmarks
                lip_distance = sum([upper_lip.y - lower_lip.y for upper_lip, lower_lip in zip(upper_lip_landmarks, lower_lip_landmarks)])

                # Store lip distance in history for smoothing
                lip_distance_history.append(lip_distance)
                if len(lip_distance_history) > 10:
                    lip_distance_history.pop(0)  # Keep a history of the last 10 frames

                # Calculate average lip distance over history
                avg_lip_distance = sum(lip_distance_history) / len(lip_distance_history)

                # Check if lips are open based on the average distance and increased threshold
                lips_open = avg_lip_distance > lip_open_threshold

                if lips_open:
                    scroll_function()  # Call your scroll function when lips are open

                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=drawing_spec)

        # Display the annotated image
        cv2.imshow('Face Mesh Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

# Release the VideoCapture and close all windows after processing
cap.release()
cv2.destroyAllWindows()
