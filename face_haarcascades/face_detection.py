import cv2
# Read file
img = cv2.imread('../face.jpg')
# Face detector object
face_detector = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
# Detect faces
faces = face_detector.detectMultiScale(img, 1.3, 5)
# Print number of faces found
print(type(faces))
face = faces[0]
# Draw rectangle around face
x, y, w, h = face
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Display image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
