'''Elizabeth Adelaide

Some deep dreaming experimentation
'''

#Main source for skeleton:
#https://pythonprogramming.net/deep-dream-python-playing-neural-network-tensorflow/

#Imports
from bin.dream import *
import numpy as np
import PIL.Image

#get layer from pre-existing model
my_file_path = "C:\\Users\\liz\\Documents\\Programs\\imageProcessing\\"
my_output_path = "C:\\Users\\liz\\Documents\\Programs\\deepDream\\outputImages\\"
layer_tensor = model.layer_tensors[3]
file_name = my_file_path + "flowers.jpg"

#load image:
img = load_image(filename='{}'.format(file_name))

#do deep dreaming
img_result = recursize_optimize(layer_tensor=layer_tensor, image=img,
    #how clear is the dream vs the original image
    num_iteration=20, step_size=1.0, rescale_factor=0.5,
    #How many passes over the data, the more passes, the more granular the gradients will be
    num_repeats=8, blend=0.2 )

img_result = np.clip(img_result, 0.0, 255.0) #clip image to prevent overflow
img_result = img_result.astype(np.uint8) #data type consistency for images

result = PIL.Image.fromarray(img_result, mode='RGB') #create image from RGB data

result.save(my_output_path+"dream.jpg")
