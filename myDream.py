'''
Elizabeth Adelaide 2018
Use facial recognition to distort faces in Images with deep dream
'''

from faceDream import model, load_image, recursive_optimize
from facialPreProcess as fPP

import numpy as np
import PIL.Image

layer_tensor = model.layer_tensors[7]

filename = 'horror.jpg'

img = load_image(filename='{}'.format('input/' + filename))

myFace = fPP.Face(filename) #init obj

face = myFace.getProcessedImage() #get processedImage

img_result, face_img_result = recursive_optimize(layer_tensor=layer_tensor, image=img, face_image=face,
     # how clear is the dream vs original image
     num_iterations=10, step_size=1.3, rescale_factor=0.5,
     # How many "passes" over the data. More passes, the more granular the gradients will be.
     num_repeats=4, blend=0.1)

img_result = np.clip(img_result, 0.0, 255.0)
img_result = img_result.astype(np.uint8)
result = PIL.Image.fromarray(img_result, mode='RGB')
result.save('output/dream_image_out.jpg')
result.show()

face_img_result = np.clip(face_img_result, 0.0, 255.0)
face_img_result = face_img_result.astype(np.uint8)
result = PIL.Image.fromarray(face_img_result, mode='RGB')
result.show()
