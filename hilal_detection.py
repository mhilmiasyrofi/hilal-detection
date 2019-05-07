#!/usr/bin/env python
from utils import *
from window import Window

import signal

import cv2
import numpy as np

import datetime

currentDT = datetime.datetime.now()
current_time = currentDT.strftime("%Y-%m-%d %H:%M:%S")

global tframe
# frame_width = 320
# frame_height = 240
# frame_width = 3096
# frame_height = 2080
frame_width = 1024
frame_height = 768

windows_name = ["raw_image", "image_stacking", "image_enhancement", "power_law", "clahe", "histogram_equalization", "fourier",
                "histogram", "blur", "canny_edge", "circle_hough_transform"]

removed = ["blur", "canny_edge", "circle_hough_transform"]

for r in removed:
    windows_name.remove(r)

MODE_POWER_LOW = 1
MODE_CLAHE = 2
MODE_HISTOGRAM_EQUALIZATION = 3
MODE_FOURIER_TRANSFORM = 4

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


def saveConfiguration(filename, ):
    # update variable value
    parameters[parameters.index("min_stack")+1] = str(min_stack)
    parameters[parameters.index("max_stack")+1] = str(max_stack)
    parameters[parameters.index(
        "image_enhancement_mode")+1] = str(image_enhancement_mode)
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

    folder = "subang"

    flat_image = None
    flat_filename = "data/video/" + folder + "/flat.jpg"
    flat_image = cv2.imread(flat_filename)

    dark_image = None
    dark_filename = "data/video/" + folder + "/dark.jpg"
    dark_image = cv2.imread(dark_filename)

    # read external file
    fh = open("parameters.txt", "r")
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
    clip_limit = int(parameters[parameters.index("clip_limit")+1])
    tile_grid_size = int(parameters[parameters.index("tile_grid_size")+1])
    blur_size = int(parameters[parameters.index("blur_size")+1])
    canny_min_val = int(parameters[parameters.index("canny_min_val")+1])
    canny_max_val = int(parameters[parameters.index("canny_max_val")+1])
    cht_min_dist = int(parameters[parameters.index("cht_min_dist")+1])
    cht_min_radius = int(parameters[parameters.index("cht_min_radius")+1])
    cht_max_radius = int(parameters[parameters.index("cht_max_radius")+1])

    ### init windows for interface
    windows = {}
    for w in windows_name:
        windows[w] = Window(w)

    ### STACKED IMAGE
    windows["image_stacking"].addTrackbar("min_stack", 0, 255, callback)
    windows["image_stacking"].setTrackbarPos("min_stack", min_stack)
    windows["image_stacking"].addTrackbar("max_stack", 0, 255, callback)
    windows["image_stacking"].setTrackbarPos("max_stack", max_stack)

    ### IMAGE ENHANCEMENT
    windows["image_enhancement"].addTrackbar(
        "image_enhancement_mode", 0, 4, callback)
    windows["image_enhancement"].setTrackbarPos(
        "image_enhancement_mode", image_enhancement_mode)

    ### IMAGE ENHANCEMENT
    windows["power_law"].addTrackbar("constant", 0, 20, callback)
    windows["power_law"].setTrackbarPos("constant", constant)
    windows["power_law"].addTrackbar("power", 0, 500, callback)
    windows["power_law"].setTrackbarPos("power", power)

    windows["clahe"].addTrackbar("clip_limit", 0, 10, callback)
    windows["clahe"].setTrackbarPos("clip_limit", clip_limit)
    windows["clahe"].addTrackbar("tile_grid_size", 0, 100, callback)
    windows["clahe"].setTrackbarPos("tile_grid_size", tile_grid_size)

    # ### BLUR
    # windows["blur"].addTrackbar("blur_size", 0, 15, callback)
    # windows["blur"].setTrackbarPos("blur_size", blur_size)

    # ### CANNY EDGE DETECTOR
    # windows["canny_edge"].addTrackbar("canny_min_val", 0, 255, callback)
    # windows["canny_edge"].addTrackbar("canny_max_val", 0, 255, callback)
    # windows["canny_edge"].setTrackbarPos("canny_min_val", canny_min_val)
    # windows["canny_edge"].setTrackbarPos("canny_max_val", canny_max_val)

    # ### CIRCLE HOUGH TRANSFORM
    # circles = []
    # windows["circle_hough_transform"].addTrackbar(
    #     "cht_min_dist", 0, 300, callback)
    # windows["circle_hough_transform"].addTrackbar(
    #     "cht_min_radius", 0, 300, callback)
    # windows["circle_hough_transform"].addTrackbar(
    #     "cht_max_radius", 0, 400, callback)
    # windows["circle_hough_transform"].setTrackbarPos(
    #     "cht_min_dist", cht_min_dist)
    # windows["circle_hough_transform"].setTrackbarPos(
    #     "cht_min_radius", cht_min_radius)
    # windows["circle_hough_transform"].setTrackbarPos(
    #     "cht_max_radius", cht_max_radius)

    camera, camera_name = initCamera(frame_width, frame_height)

    print('Enabling stills mode')
    try:
        # Force any single exposure to be halted
        camera.stop_video_capture()
        camera.stop_exposure()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass

    # warm up camera
    time.sleep(3)
    i = 0
    while (i <= 5):
        tframe = camera.capture()
        i += 1

    print('Enabling video mode')
    try:
        # Force any single exposure to be halted
        camera.stop_exposure()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass
    
    FILE_OUTPUT = 'video/' + camera_name + ' - ' +  current_time + '.avi'

    ### init video writer to save video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20, (frame_width, frame_height))

    ### init zwo asi camera
    camera.start_video_capture()

    images = []
    n_stack = 100
    sum_images = None
    mean_image = None
    removed_image = None
    try:
        i = 0
        while True:
            tframe = camera.capture_video_frame()
            if not dark_image is None:
                tframe = darkProcessor(tframe, dark_image)
            if not flat_image is None :
                tframe = flatProcessor(tframe, flat_image)
            rgb = cv2.cvtColor(tframe, cv2.COLOR_GRAY2RGB)
            out.write(rgb)
            images.append(tframe)
            if i == 0 :
                sum_images = tframe
                sum_images = sum_images.astype(np.float64)
                mean_image = tframe
                i += 1
            elif i < n_stack :
                sum_images += tframe
                mean_image = sum_images / (i+1)
                i += 1
            else :
                removed_image = images.pop(1)
                sum_images -= removed_image
                sum_images += tframe
                mean_image = sum_images / n_stack
            
            mean_image = mean_image.astype(np.uint8)
            
            # print("mean_image")
            # print(mean_image)
            # print("sum_images")
            # print(sum_images)

            # cv2.imshow("stack", mean_image)
            # cv2.imshow("raw", tframe)

            raw = tframe
            # enhanced_image = raw.copy()
            enhanced_image = mean_image.copy()

            windows["raw_image"].setImage(raw)
            windows["raw_image"].showWindow()

            windows["image_enhancement"].setImage(empty_image)
            windows["image_enhancement"].showWindow()

            min_stack = windows["image_stacking"].getTrackbarPos("min_stack")
            max_stack = windows["image_stacking"].getTrackbarPos("max_stack")

            if min_stack >= max_stack:
                min_stack = max_stack - 1
                windows["image_stacking"].setTrackbarPos("min_stack", min_stack)

            enhanced_image = np.where(
                enhanced_image > max_stack, 255, enhanced_image)
            enhanced_image = np.where(
                enhanced_image < min_stack, 0, enhanced_image)

            windows["image_stacking"].setImage(enhanced_image)
            windows["image_stacking"].showWindow()

            image_enhancement_mode = windows["image_enhancement"].getTrackbarPos(
                "image_enhancement_mode")
            windows["power_law"].setImage(empty_image)
            windows["power_law"].showWindow()
            windows["clahe"].setImage(empty_image)
            windows["clahe"].showWindow()
            windows["fourier"].setImage(empty_image)
            windows["fourier"].showWindow()
            windows["histogram_equalization"].setImage(empty_image)
            windows["histogram_equalization"].showWindow()
            if (image_enhancement_mode == MODE_POWER_LOW):
                constant = windows["power_law"].getTrackbarPos("constant")
                power = windows["power_law"].getTrackbarPos("power")
                enhanced_image = powerLawTransformation(
                    enhanced_image, constant, power)
                windows["power_law"].setImage(enhanced_image)
                windows["power_law"].showWindow()
            elif (image_enhancement_mode == MODE_CLAHE):
                clip_limit = windows["clahe"].getTrackbarPos("clip_limit")
                tile_grid_size = windows["clahe"].getTrackbarPos("tile_grid_size")
                enhanced_image = clahe(enhanced_image, clip_limit, tile_grid_size)
                windows["clahe"].setImage(enhanced_image)
                windows["clahe"].showWindow()
            elif (image_enhancement_mode == MODE_HISTOGRAM_EQUALIZATION):
                # enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
                enhanced_image = cv2.equalizeHist(enhanced_image)
                windows["power_law"].setImage(enhanced_image)
                windows["power_law"].showWindow()
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
            elif (image_enhancement_mode == MODE_FOURIER_TRANSFORM):
                enhanced_image = fourierTransform(enhanced_image)
                windows["fourier"].setImage(enhanced_image)
                windows["fourier"].showWindow()

            histogram = buildHistogramFromImage(enhanced_image)
            windows["histogram"].setImage(histogram)
            windows["histogram"].showWindow()

            # blur_size = windows["blur"].getTrackbarPos("blur_size")
            # blur = cv2.GaussianBlur(
            #     enhanced_image, (blur_size*2 + 1, blur_size*2 + 1), 0)
            # windows["blur"].setImage(blur)
            # windows["blur"].showWindow()
            
            # histogram = buildHistogramFromImage(blur)
            # windows["histogram"].setImage(histogram)
            # windows["histogram"].showWindow()

            # canny_min_val = windows["canny_edge"].getTrackbarPos("canny_min_val")
            # canny_max_val = windows["canny_edge"].getTrackbarPos("canny_max_val")
            # edge = cv2.Canny(blur, canny_min_val, canny_max_val)
            # windows["canny_edge"].setImage(edge)
            # windows["canny_edge"].showWindow()

            # cht_min_dist = windows["circle_hough_transform"].getTrackbarPos(
            #     "cht_min_dist")
            # cht_min_radius = windows["circle_hough_transform"].getTrackbarPos(
            #     "cht_min_radius")
            # cht_max_radius = windows["circle_hough_transform"].getTrackbarPos(
            #     "cht_max_radius")
            # if cht_min_dist > cht_max_radius:
            #     cht_min_radius = cht_max_radius
            #     windows["circle_hough_transform"].setTrackbarPos(
            #         "cht_min_radius", cht_min_dist)

            # circle_img = enhanced_image.copy()
            # circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, minDist=cht_min_dist,
            #                         param2=10, minRadius=cht_min_radius, maxRadius=cht_max_radius)

            # if not circles is None:
            #     circles = np.uint16(np.around(circles))
            #     for i in circles[0, :]:
            #         # draw the outer circle
            #         cv2.circle(circle_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #         # draw the center of the circle
            #         cv2.circle(circle_img, (i[0], i[1]), 2, (0, 0, 255), 3)

            # windows["circle_hough_transform"].setImage(circle_img)
            # windows["circle_hough_transform"].showWindow()


            # windows["raw_image"].moveWindow(50, 100)
            # windows["image_stacking"].moveWindow(475, 100)
            # windows["image_enhancement"].moveWindow(900, 100)
            # windows["power_law"].moveWindow(1325, 100)
            # windows["clahe"].moveWindow(1325, 100)
            # windows["histogram_equalization"].moveWindow(1325, 100)
            # windows["fourier"].moveWindow(1325, 100)

            # windows["histogram"].moveWindow(50, 550)
            # windows["blur"].moveWindow(475, 550)
            # windows["canny_edge"].moveWindow(900, 550)
            # windows["circle_hough_transform"].moveWindow(1325, 550)

            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                break
    except (KeyboardInterrupt, SystemExit):
        camera.close()
        out.release()
    
