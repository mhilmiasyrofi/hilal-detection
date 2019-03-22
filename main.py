import cv2
import numpy as np

windows_name = ["raw_image", "lhe", "blur", "canny_edge", "circle_hough_transform"]
# LHE = Local Histogram equalization
removed = ["lhe"]

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
    if img.shape[0] > 300 :
        h = img.shape[0] * 300 / img.shape[0]
        w = img.shape[1] * 300 / img.shape[0]
        img = cv2.resize(img, (int(w), int(h)))
    return img


if __name__ == "__main__":

    windows = {}
    
    for w in windows_name :
        windows[w] = Window(w)

    # im_name = "data/hilal2.jpg"
    # im_name = "data/hilal.jpg"
    im_name = "data/frame.jpg"
    img = cv2.imread(im_name, 0)
    img = resizeImage(img)
    
    ### BLUR
    blur_size = 4
    windows["blur"].addTrackbar("blur_size", 0, 15, callback)
    windows["blur"].setTrackbarPos("blur_size", blur_size)

    ### CANNY EDGE DETECTOR
    canny_min_val = 34
    canny_max_val = 159
    windows["canny_edge"].addTrackbar("canny_min_val", 0, 255, callback)
    windows["canny_edge"].addTrackbar("canny_max_val", 0, 255, callback)
    windows["canny_edge"].setTrackbarPos("canny_min_val", canny_min_val)
    windows["canny_edge"].setTrackbarPos("canny_max_val", canny_max_val)

    ### CIRCLE HOUGH TRANSFORM
    circles = []
    cht_min_dist = 20
    cht_param1 = 42
    cht_param2 = 14
    cht_min_radius = 90
    cht_max_radius = 1000
    windows["circle_hough_transform"].addTrackbar("cht_min_dist", 0, 100, callback)
    windows["circle_hough_transform"].addTrackbar("cht_param1", 0, 100, callback)
    windows["circle_hough_transform"].addTrackbar("cht_param2", 0, 100, callback)
    windows["circle_hough_transform"].addTrackbar("cht_min_radius", 0, 1000, callback)
    windows["circle_hough_transform"].addTrackbar("cht_max_radius", 0, 1000, callback)
    windows["circle_hough_transform"].setTrackbarPos("cht_min_dist", cht_min_dist)
    windows["circle_hough_transform"].setTrackbarPos("cht_param1", cht_param1)
    windows["circle_hough_transform"].setTrackbarPos("cht_param2", cht_param2)
    windows["circle_hough_transform"].setTrackbarPos("cht_min_radius", cht_min_radius)
    windows["circle_hough_transform"].setTrackbarPos("cht_max_radius", cht_max_radius)

    while (True):
        
        windows["raw_image"].setImage(img)
        windows["raw_image"].showWindow()

        blur_size = windows["blur"].getTrackbarPos("blur_size")
        blur = cv2.GaussianBlur(img, (blur_size*2 + 1, blur_size*2 + 1), 0)
        windows["blur"].setImage(blur)
        windows["blur"].showWindow()

        canny_min_val = windows["canny_edge"].getTrackbarPos("canny_min_val")
        canny_max_val = windows["canny_edge"].getTrackbarPos("canny_max_val")
        edge = cv2.Canny(blur, canny_min_val, canny_max_val)
        windows["canny_edge"].setImage(edge)  
        windows["canny_edge"].showWindow()

        cht_min_dist = windows["circle_hough_transform"].getTrackbarPos("cht_min_dist")
        cht_param1 = windows["circle_hough_transform"].getTrackbarPos("cht_param1")
        cht_param2 = windows["circle_hough_transform"].getTrackbarPos("cht_param2")
        cht_min_radius = windows["circle_hough_transform"].getTrackbarPos("cht_min_radius")
        cht_max_radius = windows["circle_hough_transform"].getTrackbarPos("cht_max_radius")
        
        circle_img = img.copy()
        circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, cht_min_dist, param1=cht_param1, param2=cht_param2, minRadius=cht_min_radius, maxRadius=cht_max_radius)
        
        if not circles is None :
            circles = np.uint16(np.around(circles))
            for i in circles[0,:] :
                # draw the outer circle
                cv2.circle(circle_img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(circle_img,(i[0],i[1]),2,(0,0,255),3)

        windows["circle_hough_transform"].setImage(circle_img)
        windows["circle_hough_transform"].showWindow()

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break



