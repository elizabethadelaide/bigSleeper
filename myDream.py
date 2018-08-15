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

layer_tensor = model.layer_tensors[4]

def processImage(filename):

    img = load_image(filename='{}'.format('input/' + filename))

    myFace = fPP.Face(filename) #init obj

    face = myFace.getProcessedImage() #get processedImage

    img_result, face_img_result = recursive_optimize(layer_tensor=layer_tensor, image=img, face_image=face,
         # how clear is the dream vs original image
         num_iterations=15, step_size=5.0, rescale_factor=0.5,
         # How many "passes" over the data. More passes, the more granular the gradients will be.
         num_repeats=2, blend=0.1)

    img_result = np.clip(img_result, 0.0, 255.0)
    img_result = img_result.astype(np.uint8)
    result = PIL.Image.fromarray(img_result, mode='RGB')
    result.save('output/' + filename)
    #result.show()

    face_img_result = np.clip(face_img_result, 0.0, 255.0)
    face_img_result = face_img_result.astype(np.uint8)
    result = PIL.Image.fromarray(face_img_result, mode='RGB')
    result.save('output/face/' + filename)
    #result.show()

totalFrames = video.video2image('ShortFaceClip.mp4') #input video

count = 0

rootdir = 'input/video'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print(os.path.join(subdir, file))
        #print('video/' + file)
        processImage('video/' + file)
        print (str(count) + '/' str(totalFrames))

img = img = load_image(filename='{}'.format('input/video/frame0.jpg'))

imgShape = img.shape

#getDimenstions, may need to flip
width = imgShape[0]
height = imgShape[1]

video.image2video('output/video', width, height, 'output/regularvideo.avi')
#just the face window
video.image2video('output/face/video', width, height, 'output/facevideo.avi')
