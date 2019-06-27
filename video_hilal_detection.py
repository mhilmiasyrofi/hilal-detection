#!/usr/bin/env python
from utils import *
from window import Window

import cv2
import numpy as np
import signal
import sys

import matplotlib.pyplot as plt

windows_name = ["raw_image", "image_stacking", "image_enhancement", "lhe", "clahe", "ghe", "power_law", "fourier",
                "histogram", "blur", "edge_detection", "circle_hough_transform"]

removed = []

for r in removed:
    windows_name.remove(r)

MODE_POWER_LOW = 1 # Power Law Operator
MODE_LHE = 2 # Local Histogram Equalization
MODE_GHE = 3 # Global Histogram Equalization
MODE_CLAHE = 4 # Clip Limit Adaptive Historgam Equalization
MODE_FOURIER_TRANSFORM = 5 # Fourier Transform

empty_image = np.zeros((1, 1, 3), np.uint8)

# Variable initiation
configuration_filename = "parameters.txt"
parameters = []
min_stack = None
max_stack = None
image_enhancement_mode = None
power = None
window_size = None
clip_limit = None
tile_grid_size = None
blur_size = None
canny_min_val = None
canny_max_val = None
cht_min_dist = None
cht_min_radius = None
cht_max_radius = None

def saveConfiguration():
    # update variable value
    parameters[parameters.index("min_stack")+1] = str(min_stack)
    parameters[parameters.index("max_stack")+1] = str(max_stack)
    parameters[parameters.index(
        "image_enhancement_mode")+1] = str(image_enhancement_mode)
    parameters[parameters.index("constant")+1] = str(constant)
    parameters[parameters.index("power")+1] = str(power)
    parameters[parameters.index("window_size")+1] = str(window_size)
    parameters[parameters.index("clip_limit")+1] = str(clip_limit)
    parameters[parameters.index("tile_grid_size")+1] = str(tile_grid_size)
    parameters[parameters.index("blur_size")+1] = str(blur_size)
    parameters[parameters.index("canny_min_val")+1] = str(canny_min_val)
    parameters[parameters.index("canny_min_val")+1] = str(canny_min_val)
    parameters[parameters.index("canny_max_val")+1] = str(canny_max_val)
    parameters[parameters.index("cht_min_dist")+1] = str(cht_min_dist)
    parameters[parameters.index("cht_min_radius")+1] = str(cht_min_radius)
    parameters[parameters.index("cht_max_radius")+1] = str(cht_max_radius)

    fw = open(configuration_filename, "w")
    [fw.write(p + "\n") for p in parameters]
    fw.close()


def signal_handler(sig, frame):
    saveConfiguration()
    sys.exit(0)

if __name__ == "__main__":

    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Not enough parameters")
        print("Usage:\nvideo_hilal_detection.py <folder> <image name>")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # read external file
    fh = open(configuration_filename, "r")
    parameters = fh.readlines()
    fh.close()
    # remove new line string
    parameters = [s.replace('\n', '') for s in parameters]

    # get each variables
    min_stack = int(parameters[parameters.index("min_stack")+1])
    max_stack = int(parameters[parameters.index("max_stack")+1])
    image_enhancement_mode = int(
        parameters[parameters.index("image_enhancement_mode")+1])
    constant = int(parameters[parameters.index("constant")+1])
    power = int(parameters[parameters.index("power")+1])
    window_size = int(parameters[parameters.index("window_size")+1])
    clip_limit = int(parameters[parameters.index("clip_limit")+1])
    tile_grid_size = int(parameters[parameters.index("tile_grid_size")+1])
    blur_size = int(parameters[parameters.index("blur_size")+1])
    canny_min_val = int(parameters[parameters.index("canny_min_val")+1])
    canny_max_val = int(parameters[parameters.index("canny_max_val")+1])
    cht_min_dist = int(parameters[parameters.index("cht_min_dist")+1])
    cht_min_radius = int(parameters[parameters.index("cht_min_radius")+1])
    cht_max_radius = int(parameters[parameters.index("cht_max_radius")+1])

    windows = {}

    for w in windows_name:
        windows[w] = Window(w)

    ### STACKED IMAGE
    windows["image_stacking"].addTrackbar("min_stack", 0, 255, callback)
    windows["image_stacking"].setTrackbarPos("min_stack", min_stack)
    windows["image_stacking"].addTrackbar("max_stack", 0, 255, callback)
    windows["image_stacking"].setTrackbarPos("max_stack", max_stack)

    ### IMAGE ENHANCEMENT
    windows["image_enhancement"].addTrackbar("image_enhancement_mode", 0, 5, callback)
    windows["image_enhancement"].setTrackbarPos("image_enhancement_mode", image_enhancement_mode)
    

    ### IMAGE ENHANCEMENT
    windows["power_law"].addTrackbar("constant", 0, 500, callback)
    windows["power_law"].setTrackbarPos("constant", constant)
    windows["power_law"].addTrackbar("power", 0, 1000, callback)
    windows["power_law"].setTrackbarPos("power", power)
   
    windows["clahe"].addTrackbar("clip_limit", 0, 100, callback)
    windows["clahe"].setTrackbarPos("clip_limit", clip_limit)
    windows["clahe"].addTrackbar("tile_grid_size", 0, 100, callback)
    windows["clahe"].setTrackbarPos("tile_grid_size", tile_grid_size)

    windows["lhe"].addTrackbar("window_size", 0, 100, callback)
    windows["lhe"].setTrackbarPos("window_size", window_size)

    ### BLUR
    blur_number = 1
    windows["blur"].addTrackbar("blur_size", 0, 15, callback)
    windows["blur"].setTrackbarPos("blur_size", blur_size)
    # windows["blur"].addTrackbar("blur_number", 0, 15, callback)
    # windows["blur"].setTrackbarPos("blur_number", blur_number)

    ### CANNY EDGE DETECTOR
    windows["edge_detection"].addTrackbar("canny_min_val", 0, 500, callback)
    windows["edge_detection"].addTrackbar("canny_max_val", 0, 500, callback)
    windows["edge_detection"].setTrackbarPos("canny_min_val", canny_min_val)
    windows["edge_detection"].setTrackbarPos("canny_max_val", canny_max_val)

    ### CIRCLE HOUGH TRANSFORM
    circles = []
    windows["circle_hough_transform"].addTrackbar("cht_min_dist", 0, 300, callback)
    windows["circle_hough_transform"].addTrackbar("cht_min_radius", 0, 300, callback)
    windows["circle_hough_transform"].addTrackbar("cht_max_radius", 0, 400, callback)
    windows["circle_hough_transform"].setTrackbarPos("cht_min_dist", cht_min_dist)
    windows["circle_hough_transform"].setTrackbarPos("cht_min_radius", cht_min_radius)
    windows["circle_hough_transform"].setTrackbarPos("cht_max_radius", cht_max_radius)

    # Create a VideoCapture object
    # folder = "data1"
    # specific_name = "video1.avi"
    folder = argv[0]
    specific_name = argv[1] + ".avi"
    # filename = "data/video/" + folder + "/hilal/" + specific_name
    filename = "/media/mhilmiasyrofi/01D1D4A5ECFA6420/TA Video Hilal/data/video/" + folder + "/hilal/" + specific_name
    cap = cv2.VideoCapture(filename)

    flat_image = None
    flat_filename = "data/video/" + folder + "/flat.jpg"
    flat_image = cv2.imread(flat_filename)

    dark_image = None
    dark_filename = "data/video/" + folder + "/dark.jpg"
    dark_image = cv2.imread(dark_filename)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))


    i = 0
    images = []
    raw_img = None
    stacked_image = None
    n_stack = 100
    while (True):
        ret, img = cap.read()

        if ret == True and i <= n_stack: 
            if not dark_image is None :
                img = darkProcessor(img, dark_image)
            if not flat_image is None :
                if not dark_image is None :
                    flat_image = darkProcessor(flat_image, dark_image)
                img = flatProcessor(img, flat_image)
            raw_img = img.copy()
            images.append(img)
            i += 1
                
        # Break the loop
        elif i == 0:
            exit()
        else :
            stacked_image = images[0].astype(np.float64)
            for i in range(1, i) :
                stacked_image += images[i]
            break
            
    releaseList(images)

    stacked_image = stacked_image/i
    stacked_image = stacked_image.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    kernel[0][0] = 0
    kernel[0][2] = 0
    kernel[2][0] = 0
    kernel[2][2] = 0

    ON = True
    OFF = False
    automatic_mode = ON
    is_done_automatic_mode = False
    if automatic_mode == ON :
        constant = 100
        power = 100
        tile_grid_size = 10
        clip_limit = 50

    while (True) :

        raw = raw_img.copy()
        raw = resizeImage(raw)

        img = stacked_image.copy()
        img = resizeImage(img)

        windows["raw_image"].setImage(img)
        windows["raw_image"].showWindow()

        min_stack = windows["image_stacking"].getTrackbarPos("min_stack")
        max_stack = windows["image_stacking"].getTrackbarPos("max_stack")

        if min_stack >= max_stack:
            min_stack = max_stack - 1
            windows["image_stacking"].setTrackbarPos("min_stack", min_stack)

        img = np.where(img > max_stack, 255, img)
        img = np.where(img < min_stack, 0, img)

        windows["image_stacking"].setImage(img)
        windows["image_stacking"].showWindow()

        windows["image_enhancement"].setImage(empty_image)
        windows["image_enhancement"].showWindow()
        
        image_enhancement_mode = windows["image_enhancement"].getTrackbarPos("image_enhancement_mode")
        windows["power_law"].setImage(empty_image)
        windows["power_law"].showWindow()
        windows["clahe"].setImage(empty_image)
        windows["clahe"].showWindow()
        windows["fourier"].setImage(empty_image)
        windows["fourier"].showWindow()
        windows["ghe"].setImage(empty_image)
        windows["ghe"].showWindow()
        windows["lhe"].setImage(empty_image)
        windows["lhe"].showWindow()
        enhanced_image = img
        if (image_enhancement_mode == MODE_POWER_LOW):
            last_constant = constant
            last_power = power
            if (automatic_mode == ON) :
                enhanced_image = powerLawTransformation(img, constant, power)
                mean = np.mean(enhanced_image)
                if mean >= 135 :
                    power += 1
                elif mean <= 120 :
                    constant += 1
                windows["power_law"].setTrackbarPos(
                    "power", power)
                windows["power_law"].setTrackbarPos(
                    "constant", constant)
            else :
                constant = windows["power_law"].getTrackbarPos("constant")
                power = windows["power_law"].getTrackbarPos("power")
                enhanced_image = powerLawTransformation(img, constant, power)
            windows["power_law"].setImage(enhanced_image)
            windows["power_law"].showWindow()
            if last_constant == constant and last_power == power :
                is_done_automatic_mode = True 
        elif (image_enhancement_mode == MODE_CLAHE) :
            last_clip_limit = clip_limit
            last_tile_grid_size = tile_grid_size
            if automatic_mode == ON :
                enhanced_image = clahe(img, clip_limit, tile_grid_size)
                mean = np.mean(enhanced_image)
                if mean >= 135:
                    if clip_limit > 10 :
                        clip_limit -= 1
                elif mean <= 120:
                    if clip_limit < enhanced_image.shape[1]:
                        clip_limit += 1
                windows["clahe"].setTrackbarPos(
                    "clip_limit", clip_limit)
                windows["clahe"].setTrackbarPos(
                    "tile_grid_size", tile_grid_size)
            else :
                clip_limit = windows["clahe"].getTrackbarPos("clip_limit")
                tile_grid_size = windows["clahe"].getTrackbarPos("tile_grid_size")
                enhanced_image = clahe(img, clip_limit, tile_grid_size)
            windows["clahe"].setImage(enhanced_image)
            windows["clahe"].showWindow()
            if last_clip_limit == clip_limit and last_tile_grid_size == tile_grid_size :
                is_done_automatic_mode = True
        elif (image_enhancement_mode == MODE_LHE):
            window_size = windows["lhe"].getTrackbarPos("window_size")
            enhanced_image = localHistogramEqualization(img, window_size)
            windows["lhe"].setImage(enhanced_image)
            windows["lhe"].showWindow()
        elif (image_enhancement_mode == MODE_GHE):
            is_done_automatic_mode = True
            enhanced_image = equalizeHistogram(img)
            windows["ghe"].setImage(enhanced_image)
            windows["ghe"].showWindow()
        elif (image_enhancement_mode == MODE_FOURIER_TRANSFORM) : 
            is_done_automatic_mode = True
            enhanced_image = fourierTransform(img)
            windows["fourier"].setImage(enhanced_image)
            windows["fourier"].showWindow()

        histogram = buildHistogramFromImage(enhanced_image)
        windows["histogram"].setImage(histogram)
        windows["histogram"].showWindow()

        blur_size = windows["blur"].getTrackbarPos("blur_size")
        blur = cv2.GaussianBlur(enhanced_image, (blur_size*2 + 1, blur_size*2 + 1), 0)
  
        windows["blur"].setImage(blur)
        windows["blur"].showWindow()

        canny_min_val = windows["edge_detection"].getTrackbarPos("canny_min_val")
        canny_max_val = windows["edge_detection"].getTrackbarPos("canny_max_val")
        # edge = sobelEdgeDetection(blur)
        # edge = cv2.Laplacian(blur, cv2.CV_64F)
        # edge, canny_min_val, canny_max_val = autoCanny(blur)
        # windows["edge_detection"].setTrackbarPos("canny_max_val",canny_max_val)
        # windows["edge_detection"].setTrackbarPos("canny_min_val",canny_min_val)
        edge = cannyEdgeDetection(blur, canny_min_val, canny_max_val)
        # edge = sobelEdgeDetection(blur)
        windows["edge_detection"].setImage(edge)
        windows["edge_detection"].showWindow()

        circle_img = enhanced_image.copy()
        if is_done_automatic_mode == True :
            cht_min_dist = windows["circle_hough_transform"].getTrackbarPos("cht_min_dist")
            cht_min_radius = windows["circle_hough_transform"].getTrackbarPos("cht_min_radius")
            cht_max_radius = windows["circle_hough_transform"].getTrackbarPos("cht_max_radius")
            if cht_min_dist > cht_max_radius :
                cht_min_radius = cht_max_radius
                windows["circle_hough_transform"].setTrackbarPos("cht_min_radius", cht_min_dist)
            
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

        windows["raw_image"].moveWindow(50, 100)
        windows["image_stacking"].moveWindow(475, 100)
        windows["image_enhancement"].moveWindow(900, 100)
        windows["fourier"].moveWindow(1325, 100)
        windows["clahe"].moveWindow(1325, 100)
        windows["power_law"].moveWindow(1325, 100)
        windows["ghe"].moveWindow(1325, 100)
        windows["lhe"].moveWindow(1325, 100)
        if (image_enhancement_mode == MODE_POWER_LOW):
            windows["power_law"].showWindow()
        elif (image_enhancement_mode == MODE_CLAHE) :
            windows["clahe"].showWindow()
        elif (image_enhancement_mode == MODE_GHE) :
            windows["ghe"].showWindow()
        elif (image_enhancement_mode == MODE_LHE) :
            windows["lhe"].showWindow()
        elif (image_enhancement_mode == MODE_FOURIER_TRANSFORM) : 
            windows["fourier"].showWindow()
        
        windows["histogram"].moveWindow(50, 550)
        windows["blur"].moveWindow(475, 550)
        windows["edge_detection"].moveWindow(900, 550)
        windows["circle_hough_transform"].moveWindow(1325, 550)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
