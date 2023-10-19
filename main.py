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
    landmarks = predictor(gray, face) # full 
    # Print type of landmarks
    print(type(landmarks))
    # Draw left eye landmarks
    cv2.circle(img, (landmarks.part(0).x, landmarks.part(0).y), 2, (0, 0, 255), 2) # left eye
    cv2.circle(img, (landmarks.part(1).x, landmarks.part(1).y), 2, (0, 0, 255), 2) # right eye
    # Draw right eye landmarks
    cv2.circle(img, (landmarks.part(2).x, landmarks.part(2).y), 2, (0, 0, 255), 2) # nose
    cv2.circle(img, (landmarks.part(3).x, landmarks.part(3).y), 2, (0, 0, 255), 2) # left mouth
    # Draw nose landmarks
    cv2.circle(img, (landmarks.part(4).x, landmarks.part(4).y), 2, (0, 0, 255), 2) # right mouth
    
    # Draw mouth landmarks
    # cv2.circle(img, (landmarks.part(5).x, landmarks.part(5).y), 2, (0, 0, 255), 2) # right mouth
    # Draw rectangle around face
    cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)


# Display image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
