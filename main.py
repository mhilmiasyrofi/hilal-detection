import cv2
import numpy as np


windows_name = ["raw_image", "lhe", "blur", "canny_edge", "hough_transform"]
# LHE = Local Histogram equalization
removed = ["lhe", "hough_transform"]

for r in removed :
    windows_name.remove(r)

def callback(x):
    pass

class Window :
    name = "Named Window"
    image = np.zeros((300, 512, 3), np.uint8)
    image[:] = [0, 0, 0]

    def __init__(self, name):
        self.name = name
        cv2.namedWindow(name)

    def setName(self, name) :
        self.name = name
    
    def getName(self) :
        return self.name
    
    def setImage(self, img) :
        self.image = img
    
    def getImage(self) :
        return self.image

    def showWindow(self):
        cv2.imshow(self.name, self.image)

    def addTrackbar(self, trackbarName, min, max, callback) :
        cv2.createTrackbar(trackbarName, self.name, min, max, callback)
    
    def getTrackbarPos(self, trackbarName):
        return cv2.getTrackbarPos(trackbarName, self.name)

    def setTrackbarPos(self, trackbarName, value):
        cv2.setTrackbarPos(trackbarName, self.name, value)
    
    def __del__(self):
        print("Window " + self.name + " is destroyed")
        cv2.destroyWindow(self.name)


def resizeImage(img) :
    if img.shape[0] > 400 :
        h = img.shape[0] * 400 / img.shape[0]
        w = img.shape[1] * 400 / img.shape[0]
        img = cv2.resize(img, (int(w), int(h)))
    return img


if __name__ == "__main__":

    windows = {}
    
    for w in windows_name :
        windows[w] = Window(w)

    # im_name = "data/hilal2.jpg"
    im_name = "data/hilal.jpg"
    img = cv2.imread(im_name, 0)
    img = resizeImage(img)
    
    canny_min_val = 100
    canny_max_val = 200
    windows["canny_edge"].addTrackbar("canny_min_val", 0, 255, callback)
    windows["canny_edge"].addTrackbar("canny_max_val", 0, 255, callback)
    windows["canny_edge"].setTrackbarPos("canny_min_val", canny_min_val)
    windows["canny_edge"].setTrackbarPos("canny_max_val", canny_max_val)
    
    while (1):

        windows["raw_image"].showWindow()

        canny_min_val = windows["canny_edge"].getTrackbarPos("canny_min_val")
        canny_max_val = windows["canny_edge"].getTrackbarPos("canny_max_val")
        edge = cv2.Canny(img, canny_min_val, canny_max_val)
        windows["canny_edge"].setImage(edge)  
        windows["canny_edge"].showWindow()

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break



