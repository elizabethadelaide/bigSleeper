'''
Elizabeth Adelaide

Uses Nose to run through tests of code
'''
import unittest
import nose2
from faceDream import model, load_image, recursive_optimize

import cv2 as cv
import numpy as np

import facialPreProcess as face

from PIL import Image
'''
Goal:

Have facial recognition run on each image and save pre-processed image
'''

imgname = 'horror.jpg'
img = load_image('input/' + imgname)

#Known working facial recognition
def preprocess(filename, fileout="face.jpg"):
    #Use Haar Cascade to Classify face
    face_cascade = cv.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')

    img = cv.imread('input/' + filename) #load image, improve string concat later
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #grayscale

    #get the faces in the image:
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #print(gray.shape)

    out = np.zeros(img.shape, dtype=np.uint8) #create blank matrix

    #for each rectangle
    for (x, y, w, h) in faces:
        #do something with the rectangles
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        subimage = img[y:y+h, x:x+w, :]
        out[y:y+h, x:x+w, :] = subimage

    #quickly display images for debugging
    #print(out.shape)
    cv.imwrite('preprocess/face.jpg', out)
    return np.float32(out) #return np array? should probably return better consistent image

myFaceObj = face.Face(imgname) #init face object with image

#test facial recognition while refactoring
#be able to get up to video recognition
class faceTest(unittest.TestCase):

    #util function
    def compareNpArrays(self, imgA, imgB):
        self.assertEqual(type(imgA), type(imgB), "Images should both np arrays")

        self.assertEqual(imgA.shape, imgB.shape, "Images should be the same shape")

        self.assertEqual(type(imgA.item(0)), type(imgB.item(0)), "Images should have the same data type")

    #self.assertEqual(imgA.item(0), imgB.item(0), "Images should have the same values")
    def isTheSameArray(self, imgA, imgB):
        for x in range(imgA.size):
            self.assertEqual(imgA.item(x), imgB.item(x))

    def test_faceReturn(self):
        myface = preprocess(imgname)

        herface = myFaceObj.getProcessedImage()

        self.compareNpArrays(myface, herface)

        self.isTheSameArray(myface, herface)
if __name__ == '__main__':

    unittest.main()
