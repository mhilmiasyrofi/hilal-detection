import cv2
import numpy as np
from matplotlib import pyplot as plt
import signal
import sys

from matplotlib import pyplot as plt

windows_name = ["raw_image", "image_stacking", "image_enhancement", "clahe", "power_law", "fourier",
                "histogram", "blur", "canny_edge", "circle_hough_transform"]

removed = []

for r in removed:
    windows_name.remove(r)

MODE_POWER_LOW = 1
MODE_CLAHE = 2
MODE_FOURIER_TRANSFORM = 3
MODE_GAMMA_CORRECTION = 4

empty_image = np.zeros((1, 1, 3), np.uint8)

# Variable initiation
parameters = []
min_stack = None
max_stack = None
image_enhancement_mode = None
power = None
clip_limit = None
tile_grid_size = None
blur_size = None
canny_min_val = None
canny_max_val = None
cht_min_dist = None
cht_min_radius = None
cht_max_radius = None


def callback(x):
    pass


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

    def moveWindow(self, x, y) :
        cv2.moveWindow(self.name, x, y)

    def __del__(self):
        print("Window " + self.name + " is destroyed")
        cv2.destroyWindow(self.name)


def resizeImage(img):
    if img.shape[0] > 300:
        h = img.shape[0] * 300 / img.shape[0]
        w = img.shape[1] * 300 / img.shape[0]
        img = cv2.resize(img, (int(w), int(h)))
    return img


def convertFloat64ImgtoUint8(img):
    # Get the information of the incoming image type
    # normalize the img to 0 - 1
    img = img.astype(np.float64) / float(img.max())
    img = 255 * img  # Now scale by 255
    img = img.astype(np.uint8)
    return img


def powerLawTransformation(img, constant=10, power=100):
    power_law = img.astype(np.float64)
    power_law = power_law/255
    power_law = cv2.pow(power_law, power/100)
    power_law = convertFloat64ImgtoUint8(power_law)
    power_law = power_law * float(constant/10)
    power_law = np.where(power_law > 254, 254, power_law)
    power_law = np.where(power_law < 0, 0, power_law)
    power_law = power_law.astype(np.uint8)
    return power_law


def clahe(img, clipLimit=2.0, tileGridSize=8):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (tileGridSize == 0):
        tileGridSize = 1
    clahe = cv2.createCLAHE(clipLimit, (tileGridSize, tileGridSize))
    img = clahe.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.where(img > 254, 254, img)
    img = np.where(img < 0, 0, img)
    img = img.astype(np.uint8)
    return img


def fourierTransform(img):
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Convert back again
    # normalize the data to 0 - 1
    img_back = img_back.astype(np.float32) / img_back.max()
    img_back = 255 * img_back  # Now scale by 255
    img = img_back.astype(np.uint8)

    # print(img.shape)
    # print("img")
    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print(img.shape)

    return img


def release_list(l):
    del l[:]
    del l


def darkProcessor(img, dark):
    img = img - dark
    img = img.astype(np.uint8)
    return img


def flatProcessor(img, flat):
    img = img/flat
    img *= 255
    img = img.astype(np.uint8)
    return img


def buildHistogramFromImage(image):
    h = np.zeros((300, 256, 3))
    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([image], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
    h = np.flipud(h)
    return h


def saveConfiguration(filename) :
    # update variable value
    parameters[parameters.index("min_stack")+1] = str(min_stack)
    parameters[parameters.index("max_stack")+1] = str(max_stack)
    parameters[parameters.index("image_enhancement_mode")+1] = str(image_enhancement_mode)
    parameters[parameters.index("constant")+1] = str(constant)
    parameters[parameters.index("power")+1] = str(power)
    parameters[parameters.index("clip_limit")+1] = str(clip_limit)
    parameters[parameters.index("tile_grid_size")+1] = str(tile_grid_size)
    parameters[parameters.index("blur_size")+1] = str(blur_size)
    parameters[parameters.index("canny_min_val")+1] = str(canny_min_val)
    parameters[parameters.index("canny_min_val")+1] = str(canny_min_val)
    parameters[parameters.index("canny_max_val")+1] = str(canny_max_val)
    parameters[parameters.index("cht_min_dist")+1] = str(cht_min_dist)
    parameters[parameters.index("cht_min_radius")+1] = str(cht_min_radius)
    parameters[parameters.index("cht_max_radius")+1] = str(cht_max_radius)

    fw = open(filename, "w")
    [fw.write(p + "\n") for p in parameters]
    fw.close()

def signal_handler(sig, frame):
    saveConfiguration("parameters.txt")
    sys.exit(0)

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    
    windows = {}

    for w in windows_name:
        windows[w] = Window(w)

    # Create a VideoCapture object
    folder = "data3"
    specific_name = "video1.avi"
    filename = "data/video/" + folder + "/hilal/" + specific_name
    cap = cv2.VideoCapture(filename)

    flat_image = None
    flat_filename = "data/video/" + folder + "/flat/flat.jpg" 
    flat_image = cv2.imread(flat_filename)
    
    
    dark_image = None
    dark_filename = "data/video/" + folder + "/dark/dark.jpg" 
    dark_image = cv2.imread(dark_filename)
    

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # read external file
    fh = open("parameters.txt", "r")
    parameters = fh.readlines()
    fh.close()
    # remove new line string
    parameters = [s.replace('\n', '') for s in parameters]

    # get each variables
    min_stack = int(parameters[parameters.index("min_stack")+1])
    max_stack = int(parameters[parameters.index("max_stack")+1])
    image_enhancement_mode = int(parameters[parameters.index("image_enhancement_mode")+1])
    constant = int(parameters[parameters.index("constant")+1])
    power = int(parameters[parameters.index("power")+1])
    clip_limit = int(parameters[parameters.index("clip_limit")+1])
    tile_grid_size = int(parameters[parameters.index("tile_grid_size")+1])
    blur_size = int(parameters[parameters.index("blur_size")+1])
    canny_min_val = int(parameters[parameters.index("canny_min_val")+1])
    canny_max_val = int(parameters[parameters.index("canny_max_val")+1])
    cht_min_dist = int(parameters[parameters.index("cht_min_dist")+1])
    cht_min_radius = int(parameters[parameters.index("cht_min_radius")+1])
    cht_max_radius = int(parameters[parameters.index("cht_max_radius")+1])

    ### STACKED IMAGE
    windows["image_stacking"].addTrackbar("min_stack", 0, 255, callback)
    windows["image_stacking"].setTrackbarPos("min_stack", min_stack)
    windows["image_stacking"].addTrackbar("max_stack", 0, 255, callback)
    windows["image_stacking"].setTrackbarPos("max_stack", max_stack)

    ### IMAGE ENHANCEMENT
    windows["image_enhancement"].addTrackbar("image_enhancement_mode", 0, 3, callback)
    windows["image_enhancement"].setTrackbarPos("image_enhancement_mode", image_enhancement_mode)
    

    ### IMAGE ENHANCEMENT
    windows["power_law"].addTrackbar("constant", 0, 20, callback)
    windows["power_law"].setTrackbarPos("constant", constant)
    windows["power_law"].addTrackbar("power", 0, 500, callback)
    windows["power_law"].setTrackbarPos("power", power)
   
    windows["clahe"].addTrackbar("clip_limit", 0, 10, callback)
    windows["clahe"].setTrackbarPos("clip_limit", clip_limit)
    windows["clahe"].addTrackbar("tile_grid_size", 0, 100, callback)
    windows["clahe"].setTrackbarPos("tile_grid_size", tile_grid_size)

    ### BLUR
    windows["blur"].addTrackbar("blur_size", 0, 15, callback)
    windows["blur"].setTrackbarPos("blur_size", blur_size)

    ### CANNY EDGE DETECTOR
    windows["canny_edge"].addTrackbar("canny_min_val", 0, 255, callback)
    windows["canny_edge"].addTrackbar("canny_max_val", 0, 255, callback)
    windows["canny_edge"].setTrackbarPos("canny_min_val", canny_min_val)
    windows["canny_edge"].setTrackbarPos("canny_max_val", canny_max_val)

    ### CIRCLE HOUGH TRANSFORM
    circles = []
    windows["circle_hough_transform"].addTrackbar("cht_min_dist", 0, 300, callback)
    windows["circle_hough_transform"].addTrackbar("cht_min_radius", 0, 300, callback)
    windows["circle_hough_transform"].addTrackbar("cht_max_radius", 0, 400, callback)
    windows["circle_hough_transform"].setTrackbarPos("cht_min_dist", cht_min_dist)
    windows["circle_hough_transform"].setTrackbarPos("cht_min_radius", cht_min_radius)
    windows["circle_hough_transform"].setTrackbarPos("cht_max_radius", cht_max_radius)
    
    i = 0
    images = []
    raw_img = None
    stacked_image = None
    while (True):
        ret, img = cap.read()

        if ret == True and i <= 100: 
            if not dark_image is None :
                # uncalibrated = resizeImage(img.copy())
                img = darkProcessor(img, dark_image)
                # cv2.imshow("uncalibrated", uncalibrated)
                # cv2.imshow("dark", resizeImage(dark_image))
            if not flat_image is None :
                img = flatProcessor(img, flat_image)
            raw_img = img.copy()
            # windows["raw_image"].setImage(img)
            # windows["raw_image"].showWindow()
            enhanced_image = img.copy()

            images.append(img)
            i += 1
                
        # Break the loop
        elif i == 0:
            exit()
        else :
            # print(len(images))
            stacked_image = images[0].astype(np.float64)
            for i in range(1, i) :
                stacked_image += images[i]
            break
            
    release_list(images)

    stacked_image = stacked_image/i
    stacked_image = stacked_image.astype(np.uint8)
    
    while (True) :
        img = stacked_image.copy()
        enhanced_image = resizeImage(img)
        raw = raw_img.copy()
        raw = resizeImage(raw)

        windows["raw_image"].setImage(raw)
        windows["raw_image"].showWindow()
        
        windows["image_enhancement"].setImage(empty_image)
        windows["image_enhancement"].showWindow()

        min_stack = windows["image_stacking"].getTrackbarPos("min_stack")
        max_stack = windows["image_stacking"].getTrackbarPos("max_stack")
        
        if min_stack >= max_stack :
            min_stack = max_stack - 1
            windows["image_stacking"].setTrackbarPos("min_stack", min_stack)

        enhanced_image = np.where(enhanced_image > max_stack, 255, enhanced_image)
        enhanced_image = np.where(enhanced_image < min_stack, 0, enhanced_image)

        windows["image_stacking"].setImage(enhanced_image)
        windows["image_stacking"].showWindow()
        
        image_enhancement_mode = windows["image_enhancement"].getTrackbarPos("image_enhancement_mode")
        if (image_enhancement_mode == MODE_POWER_LOW):
            constant = windows["power_law"].getTrackbarPos("constant")
            power = windows["power_law"].getTrackbarPos("power")
            enhanced_image = powerLawTransformation(enhanced_image, constant, power)
            windows["power_law"].setImage(enhanced_image)
            windows["power_law"].showWindow()
            windows["clahe"].setImage(empty_image)
            windows["clahe"].showWindow()
            windows["fourier"].setImage(empty_image)
            windows["fourier"].showWindow()
        elif (image_enhancement_mode == MODE_CLAHE) :
            clip_limit = windows["clahe"].getTrackbarPos("clip_limit")
            tile_grid_size = windows["clahe"].getTrackbarPos("tile_grid_size")
            enhanced_image = clahe(enhanced_image, clip_limit, tile_grid_size)
            windows["clahe"].setImage(enhanced_image)
            windows["clahe"].showWindow()
            windows["power_law"].setImage(empty_image)
            windows["power_law"].showWindow()
            windows["fourier"].setImage(empty_image)
            windows["fourier"].showWindow()
        elif (image_enhancement_mode == MODE_FOURIER_TRANSFORM) : 
            enhanced_image = fourierTransform(enhanced_image)
            windows["fourier"].setImage(enhanced_image)
            windows["fourier"].showWindow()
            windows["clahe"].setImage(empty_image)
            windows["clahe"].showWindow()
            windows["power_law"].setImage(empty_image)
            windows["power_law"].showWindow()
        else :
            windows["power_law"].setImage(empty_image)
            windows["power_law"].showWindow()
            windows["clahe"].setImage(empty_image)
            windows["clahe"].showWindow()
            windows["fourier"].setImage(empty_image)
            windows["fourier"].showWindow()
    

        blur_size = windows["blur"].getTrackbarPos("blur_size")
        blur = cv2.GaussianBlur(enhanced_image, (blur_size*2 + 1, blur_size*2 + 1), 0)
        windows["blur"].setImage(blur)
        windows["blur"].showWindow()

        histogram = buildHistogramFromImage(blur)
        windows["histogram"].setImage(histogram)
        windows["histogram"].showWindow()

        canny_min_val = windows["canny_edge"].getTrackbarPos("canny_min_val")
        canny_max_val = windows["canny_edge"].getTrackbarPos("canny_max_val")
        edge = cv2.Canny(blur, canny_min_val, canny_max_val)
        windows["canny_edge"].setImage(edge)
        windows["canny_edge"].showWindow()

        cht_min_dist = windows["circle_hough_transform"].getTrackbarPos("cht_min_dist")
        cht_min_radius = windows["circle_hough_transform"].getTrackbarPos("cht_min_radius")
        cht_max_radius = windows["circle_hough_transform"].getTrackbarPos("cht_max_radius")
        if cht_min_dist > cht_max_radius :
            cht_min_radius = cht_max_radius
            windows["circle_hough_transform"].setTrackbarPos("cht_min_radius", cht_min_dist)
        

        circle_img = enhanced_image.copy()
        circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, minDist=cht_min_dist, param2=10 ,minRadius=cht_min_radius, maxRadius=cht_max_radius)

        if not circles is None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:] :
                # draw the outer circle
                cv2.circle(circle_img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(circle_img,(i[0],i[1]),2,(0,0,255),3)

        windows["circle_hough_transform"].setImage(circle_img)
        windows["circle_hough_transform"].showWindow()

        # windows_name = ["raw_image", "image_stacking", "image_enhancement", "clahe", "power_law", "fourier",
        #     "blur", "canny_edge", "circle_hough_transform"]

        windows["raw_image"].moveWindow(50, 100)
        windows["image_stacking"].moveWindow(475, 100)
        windows["image_enhancement"].moveWindow(900, 100)
        if (image_enhancement_mode == MODE_POWER_LOW):
            windows["fourier"].moveWindow(1325, 100)
            windows["clahe"].moveWindow(1325, 100)
            windows["power_law"].moveWindow(1325, 100)
        elif (image_enhancement_mode == MODE_CLAHE) :
            windows["clahe"].moveWindow(1325, 100)
            windows["power_law"].moveWindow(1325, 100)
            windows["fourier"].moveWindow(1325, 100)
        elif (image_enhancement_mode == MODE_FOURIER_TRANSFORM) : 
            windows["fourier"].moveWindow(1325, 100)
            windows["power_law"].moveWindow(1325, 100)
            windows["clahe"].moveWindow(1325, 100)
        

        windows["histogram"].moveWindow(50, 550)
        windows["blur"].moveWindow(475, 550)
        windows["canny_edge"].moveWindow(900, 550)
        windows["circle_hough_transform"].moveWindow(1325, 550)


        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            breako
