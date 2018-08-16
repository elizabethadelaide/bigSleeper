# bigSleeper
Deep Dreaming Experimenting in Python

Uses tensorflow and opencv to create deep dream images and videos with facial recognition. Uses nose2 and unittest for testdriven development for some aspects of the program. Uses Inception5h as deep dream training data.

Basic Deep Dream image, from a space image

![Flowers](images/basicDeepDream.jpeg)

# Image processing

Each image is first processes using a Haas filter with for both frontal and profile facial features. The detected facial rectangles are isolated and saved into separated images.

![Just a face](preprocess/face.jpg)

That face image is used for the deep dream algorithm to calculate the gradient of the image. After getting the optimized value based on the gradient and the layer tensor, apply the value to the original image. This will cause the algorithm to respond to values within the facial box, distorting facial features. The algorithm will be agnostic to features outside of the facial rectangle.

![A very creepy image](images/dream_image_out.jpg)

Playing with the options a bit can get more horrifying

![Scary](images/scary.jpg)

# Video

Videos are made by processing each image indepedently as a sequence.

[![Link to youtube for clip](http://img.youtube.com/vi/C7i4bdkKbvE/0.jpg)](http://www.youtube.com/watch?v=C7i4bdkKbvE "Basic Clip of Deep Dreamer")

Also generated is the window of the facial recognition, which has interesting results.

[![Link to youtube for facial window](http://img.youtube.com/vi/T51XGUskI3Y/0.jpg)](http://www.youtube.com/watch?v=T51XGUskI3Y "Facial Window")
