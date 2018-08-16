'''
Elizabeth Adelaide 2018
Use facial recognition to distort faces in Images with deep dream
'''

from faceDream import model, load_image, recursive_optimize
import facialPreProcess as fPP

import os

import numpy as np
import PIL.Image

import video

layer_tensor = model.layer_tensors[11] #Which layer model to use

def processImage(filename):

    img = load_image(filename='{}'.format('input/' + filename))

    myFace = fPP.Face(filename) #init obj, do processing

    face = myFace.getProcessedImage() #get processedImage, just isolated faces

    img_result, face_img_result = recursive_optimize(layer_tensor=layer_tensor, image=img, face_image=face,
         # how clear is the dream vs original image
         num_iterations=2, step_size=10.0, rescale_factor=0.9,
         # How many "passes" over the data. More passes, the more granular the gradients will be.
         num_repeats=10, blend=0.5)

    #Save full image
    img_result = np.clip(img_result, 0.0, 255.0)
    img_result = img_result.astype(np.uint8)
    result = PIL.Image.fromarray(img_result, mode='RGB')
    result.save('output/' + filename)
    #result.show()

    #Save distorted facial image, because I like it
    face_img_result = np.clip(face_img_result, 0.0, 255.0)
    face_img_result = face_img_result.astype(np.uint8)
    result = PIL.Image.fromarray(face_img_result, mode='RGB')
    result.save('output/face/' + filename)
    #result.show()

def processVideo(videoName):
    totalFrames = video.video2image(videoName) #input video

    print("Loaded video with %d frames" % totalFrames)

    count = 0 #For progression tracking

    rootdir = 'input/video'
    for subdir, dirs, files in os.walk(rootdir): #For each frame in directory, note, this won't be linear
        for file in files:
            #print(os.path.join(subdir, file))
            print('video/' + file)
            processImage('video/' + file)
            print (str(count) + '/' + str(totalFrames))  #Display progress
            count = count + 1

    img = load_image(filename='{}'.format('input/video/frame0.jpg'))

    imgShape = img.shape

    #getDimenstions, may need to flip
    width = imgShape[0]
    height = imgShape[1]

    video.image2video('output/video', width, height, 'output/regularvideo.avi')
    #just the face window
    video.image2video('output/face/video', width, height, 'output/facevideo.avi')

processVideo('Chaplin.avi') #process a video
