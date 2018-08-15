'''
Runs facial detection through opencv
Creates clipping mask and outputs boxes of faces
'''

import numpy as np
import cv2 as cv

'''
I have an idea that I will be chainging deep dream algorithm
Right now I'm not fully happy with how it works
But want to see it with a video and then rework it
'''

#class contains face information for an image
#stores image info, and individual rectangles of each face
class Face:

    def __init__(self, imgname):
        self.imgname = 'input/' + imgname
        self.outname = 'preprocess/' + imgname
        self.preprocess()

    #change the image to process
    def process(self, imgname):
        self.imgname = 'input/' + imgname
        self.outname = 'preprocess/' + imgname

        self.preprocess()

    #Getters:
    def getOutname(self):
        return self.outname
    def getProcessedImage(self):
        return self.processedImage

    def preprocess(self):

        img = cv.imread(self.imgname)

        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) #grayscale

        self.out = np.zeros(img.shape, dtype=np.uint8) #create blank matrix

        #Checks for front and profile face
        self.runProcesser(img, gray, 'opencv/haarcascade_frontalface_default.xml')
        self.runProcesser(img, gray, 'opencv/haarcascade_profileface')

        #quickly display images for debugging
        #print(out.shape)
        cv.imwrite(self.outname, self.out)
        self.processedImage = np.float32(self.out) #return np array? should probably return better consistent image

    #TODO: Add processing to get subimages of image
    #for each face, return x,y,w,h and ndarrays of subimages
    #will have to see if that will be helpful

    #Main processing function, internal:
    def runProcesser(self, img, gray, processName):

        #Use Haar Cascade to Classify face
        face_cascade = cv.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')


        #get the faces in the image:
        faces = face_cascade.detectMultiScale(gray, 1.3, 2)

        #print(gray.shape)


        #for each rectangle
        for (x, y, w, h) in faces:
            #do something with the rectangles
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            subimage = img[y:y+h, x:x+w, :]
            self.out[y:y+h, x:x+w, :] = subimage
