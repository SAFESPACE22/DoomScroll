import cv2
import mediapipe as mp
from autoScroll import scroll_function

# Initialize drawing utilities and face mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize video capture
cap = cv2.VideoCapture(0)

# Drawing specification for irises
iris_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Create a FaceMesh object with refine_landmarks enabled for iris detection
with mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Set to False for video processing
    max_num_faces=1,          # Process only one face at a time
    refine_landmarks=True,    # Enables iris detection
    min_detection_confidence=0.5) as face_mesh:

    def iris_moved_down(current_pos, previous_pos, threshold=0.01):
        """ Method to check if iris moved down significantly """
        return (current_pos - previous_pos) > threshold

    previous_iris_y = None
    y_positions = []

    # Loop to read from the video capture
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(img_rgb)
        
        # Draw only the iris landmarks on the image
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,  # Only draw iris connections
                    landmark_drawing_spec=iris_drawing_spec,
                    connection_drawing_spec=iris_drawing_spec)
                
                # Get iris landmarks (for left and right eye)
                left_iris = face_landmarks.landmark[468:473]  # Left iris landmarks
                right_iris = face_landmarks.landmark[473:478] # Right iris landmarks

                # Calculate the average y position of the irises
                left_iris_y = sum([pt.y for pt in left_iris]) / len(left_iris)
                right_iris_y = sum([pt.y for pt in right_iris]) / len(right_iris)
                
                current_iris_y = (left_iris_y + right_iris_y) / 2  # Average y position of both irises

                y_positions.append(current_iris_y)
                if len(y_positions) > 5:  # Use the last 5 positions to calculate a moving average
                    y_positions.pop(0)
                avg_iris_y = sum(y_positions) / len(y_positions)

                if previous_iris_y is not None:
                    if iris_moved_down(avg_iris_y, previous_iris_y):
                        scroll_function()

                previous_iris_y = avg_iris_y
        
        # Display the annotated image
        cv2.imshow('MediaPipe Iris Detection', img)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
