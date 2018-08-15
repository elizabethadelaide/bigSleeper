'''
Runs facial detection through opencv
Creates clipping mask and outputs boxes of faces
'''

import numpy as np
import cv2 as cv

'''
Start with hardcoded image
Later rewrite as function
'''

#Use Haar Cascade to Classify face
face_cascade = cv.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')

img = cv.imread('input/horror.jpg') #load image
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #grayscale

#get the faces in the image:
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(gray.shape)

out = np.zeros(img.shape, dtype=np.uint8) #create blank matrix

#for each rectangle
for (x, y, w, h) in faces:
    #do something with the rectangles
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    subimage = img[y:y+h, x:x+w, :]
    out[y:y+h, x:x+w, :] = subimage

#quickly display images for debugging
print(out.shape)
cv.imwrite('preprocess/face.jpg', out)
print("Written out")