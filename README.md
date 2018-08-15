# bigSleeper
Deep Dreaming Experimenting in Python

In progress. Contains utilities in bin/deepDream.py. Testing-driven development being built using nose. Tensorflow and opencv as main libraries.

Basic Deep Dream image, from a space image

![Flowers](images/basicDeepDream.jpeg)

# Still Images

Current concept for this project is to particularly target faces in images, and to create horror images, and eventually short videos. The current flow is to preprocess images and isolate faces in images using a opencv Haar filter:

![Just a face](preprocess/face.jpg)

Then use that face image as the input for the deep dream algorithm to calculate the gradient of the image. After getting the optimized value based on the gradient and the layer tensor, apply the value to the original image. This will cause the algorithm to respond to values within the facial box, distorting facial features. Outside the facial boxes, the algorithm will be agnostic to features.

![A very creepy image](images/dream_image_out.jpg)

Playing with the options a bit can get more horrifying

![Scary](images/scary.jpg)

# Video

Videos are made by processing each image independenently as a sequence.

[![Link to youtube](http://img.youtube.com/vi/https://youtu.be/C7i4bdkKbvE/0.jpg)](http://www.youtube.com/watch?v=https://youtu.be/C7i4bdkKbvE "Example clip")

Also generated is the window of the facial recognition, which has interesting results.

[![Link to youtube](http://img.youtube.com/vi/https://youtu.be/https://youtu.be/T51XGUskI3Y/0.jpg)](http://www.youtube.com/watch?v=https://youtu.be/https://youtu.be/T51XGUskI3Y "Example clip")
