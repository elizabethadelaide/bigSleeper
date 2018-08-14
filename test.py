'''
Elizabeth Adelaide

Uses Nose to run through tests of code
'''
import unittest
from bin.dream import * #Deep dream starting code

from PIL import Image
'''
Run through Pedersen's deep dream code and assert expected values

Allow for quick refactoring
'''
class dreamerQuickStartTest(unittest.TestCase):
    //initialize with some sample image values
    def __init__(self):
        self.myPhotoPath = "C:\Users\liz\Documents\Programs\imageProcessing"

    //try loading an image
    def test_load_image(self):
        myPhoto = self.myPhotoPath + "\flowers.jpg"
        loadFile = load_image(myPhoto) #function to test

        jpgFile = Image.open(myPhoto)
        assertTrue((jpgFile.width == loadFile.width) && (jpgFile.height && loadFile.height))



if __name__ == '__main__':
    unittest.main()
