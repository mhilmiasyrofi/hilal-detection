import cv2
import numpy as np

class Window:
    name = "Named Window"
    image = np.zeros((300, 512, 3), np.uint8)
    image[:] = [0, 0, 0]

    def __init__(self, name):
        self.name = name
        cv2.namedWindow(name)

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def setImage(self, img):
        self.image = img

    def getImage(self):
        return self.image

    def showWindow(self):
        cv2.imshow(self.name, self.image)

    def deleteWindow(self):
        cv2.destroyWindow(self.name)

    def addTrackbar(self, trackbarName, min, max, callback):
        cv2.createTrackbar(trackbarName, self.name, min, max, callback)

    def getTrackbarPos(self, trackbarName):
        return cv2.getTrackbarPos(trackbarName, self.name)

    def setTrackbarPos(self, trackbarName, value):
        cv2.setTrackbarPos(trackbarName, self.name, value)

    def moveWindow(self, x, y):
        cv2.moveWindow(self.name, x, y)

    def __del__(self):
        # print("Window " + self.name + " is destroyed")
        cv2.destroyWindow(self.name)
