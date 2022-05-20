import cv2
from random import randrange

from numpy import True_

# Load some preinstalled data on face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img = cv2.imread('rdj_img.webp')
# img = cv2.imread('practice.jpg')
# img = cv2.imread('pre.jpg')
webcam = cv2.VideoCapture(0)

# Iterate forerver over frames
while True:

    # Read the current frame
    successfully_read_frame, frame = webcam.read()

    # Convert image to greyscale
    grayScaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    face_coordinates = trained_face_data.detectMultiScale(grayScaled_img)

    # Draw rectangle around faces
    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Pradip Face Detector : ', frame)

    # Pause the Program until we enter any key
    key = cv2.waitKey(1)

    # If Q is pressed then stop camera
    if key == 81 or key == 113:
        break

webcam.release()

print("Code Completed")
