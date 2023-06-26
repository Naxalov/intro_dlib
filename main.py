import cv2
import dlib

# Frontal face detector
detector = dlib.get_frontal_face_detector()
# print type of detector
print(type(detector))
# Predictor for 5 face landmarks
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# Read image
img = cv2.imread("face.jpg")
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

# Loop through faces
for face in faces:
    # Get face landmarks
    landmarks = predictor(gray, face)
    # Print type of landmarks
    print(type(landmarks))
    # Loop through landmarks
    for n in range(0, 5):
        # Get x and y coordinates
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        # Draw a circle
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # Draw rectangle around face
    cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)


# Display image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
