import cv2
import dlib

# Frontal face detector
detector = dlib.get_frontal_face_detector()
# Read image
img = cv2.imread('face.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Use detector to find landmarks
faces = detector(gray)
# Loop through faces
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
    # Draw a rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Show the face
cv2.imshow("Face", img)
# Wait for a key press to exit
cv2.waitKey(0)
# Destroy all windows
cv2.destroyAllWindows()
   
