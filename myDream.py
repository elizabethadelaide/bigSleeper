from faceDream import model, load_image, recursive_optimize
import numpy as np
import PIL.Image

layer_tensor = model.layer_tensors[5]
img = load_image(filename='{}'.format('input/horror.jpg'))

face = load_image(filename='{}'.format('preprocess/face.jpg'))

img_result = recursive_optimize(layer_tensor=layer_tensor, image=img, face_image=face,
     # how clear is the dream vs original image
     num_iterations=15, step_size=1.6, rescale_factor=0.84,
     # How many "passes" over the data. More passes, the more granular the gradients will be.
     num_repeats=6, blend=0.2)

img_result = np.clip(img_result, 0.0, 255.0)
img_result = img_result.astype(np.uint8)
result = PIL.Image.fromarray(img_result, mode='RGB')
result.save('output/dream_image_out.jpg')
result.show()
