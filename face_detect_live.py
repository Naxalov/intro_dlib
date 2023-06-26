import cv2
import dlib

# Frontal face detector
detector = dlib.get_frontal_face_detector()
# print type of detector
print(type(detector))
# Predictor for 5 face landmarks
# predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# Predictor for 68 face landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get video feed from webcam
cap = cv2.VideoCapture(0)

# Loop until user presses 'q'
while True:
    # Read frame
    _, frame = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Loop through faces
    for face in faces:
        # Get face landmarks
        landmarks = predictor(gray, face)
        # Print type of landmarks
        # Loop through landmarks
        for n in range(0, 64):
            # Get x and y coordinates
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # Draw a circle
            # Color for landmark
            color = (0,255,0)
            cv2.circle(frame, (x, y), 2, color, -1)
        # Draw rectangle around face
        # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Output", frame)

    # Check if user pressed 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# Release video capture object
cap.release()
# Destroy all windows
cv2.destroyAllWindows()

