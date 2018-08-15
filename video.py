'''
Handle video frames

May have to reinstall opencv from source

I know installing from pip has issues with video
'''

import cv2 as cv
import os

def video2image(videoName):
    #Again, start as hardcode and rewrite as functions
    #Should I clip the video here? yes
    #Avoid as much external video processing as possible
    vidcap = cv.VideoCapture('input/' + videoName)
    success, image = vidcap.read()
    count = 0
    directory = 'input/video/'
    while success:
        filename = directory + "frame" + str(count) + ".jpg"
        cv.imwrite(filename, image)
        success, image = vidcap.read()
        count += 1
        print("Frame %d" % count)
    print("Wrote %d to file" % count)

def image2video(rootdir, width, height, videoName):
    video = cv.VideoWriter(videoName, -1, 1, (width, height))
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print(os.path.join(subdir, file))
            img = cv.imread(os.path.join(subdir, file))
            video.write(img)
    cv.destroyAllWindows()
    video.release()
