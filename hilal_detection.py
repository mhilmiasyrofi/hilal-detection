#!/usr/bin/env python

from matplotlib import pyplot as plt
import signal
import argparse
import os
import sys
import time
import zwoasi as asi

import cv2
import numpy as np

import datetime

currentDT = datetime.datetime.now()
current_time = currentDT.strftime("%Y-%m-%d %H:%M:%S")

global tframe
frame_width = 320
frame_height = 240
# frame_width = 1280
# frame_height = 960

def initCamera(cam_width=320, cam_height=240):

    env_filename = os.getenv('ZWO_ASI_LIB')

    parser = argparse.ArgumentParser(
        description='Crescent moon detector')
    parser.add_argument('filename',
                        nargs='?',
                        help='SDK library filename')
    args = parser.parse_args()

    # Initialize zwoasi with the name of the SDK library
    if args.filename:
        asi.init(args.filename)
    elif env_filename:
        asi.init(env_filename)
    else:
        print('The filename of the SDK library is required (or set ZWO_ASI_LIB environment variable with the filename)')
        sys.exit(1)

    num_cameras = asi.get_num_cameras()
    if num_cameras == 0:
        print('No cameras found')
        sys.exit(0)

    cameras_found = asi.list_cameras()  # Models names of the connected cameras

    if num_cameras == 1:
        camera_id = 0
        camera_name = cameras_found[0]
        print('Found one camera: %s' % cameras_found[0])
    else:
        print('Found %d cameras' % num_cameras)
        for n in range(num_cameras):
            print('    %d: %s' % (n, cameras_found[n]))
        # TO DO: allow user to select a camera
        camera_id = 0
        print('Using #%d: %s' % (camera_id, cameras_found[camera_id]))

    camera = asi.Camera(camera_id)
    camera_info = camera.get_camera_property()

    # Get all of the camera controls
    controls = camera.get_controls()

    # Use minimum USB bandwidth permitted
    camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD,
                            camera.get_controls()['BandWidth']['MinValue'])

    # Set some sensible defaults. They will need adjusting depending upon
    # the sensitivity, lens and lighting conditions used.
    camera.disable_dark_subtract()

    camera.set_control_value(asi.ASI_GAIN, 150)
    camera.set_control_value(asi.ASI_EXPOSURE, 100)
    camera.set_control_value(asi.ASI_WB_B, 99)
    camera.set_control_value(asi.ASI_WB_R, 75)
    camera.set_control_value(asi.ASI_GAMMA, 50)
    camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
    camera.set_control_value(asi.ASI_FLIP, 0)

    # Restore all controls to default values except USB bandwidth
    for c in controls:
        if controls[c]['ControlType'] == asi.ASI_BANDWIDTHOVERLOAD:
            continue
        camera.set_control_value(
            controls[c]['ControlType'], controls[c]['DefaultValue'])

    # Can autoexposure be used?
    k = 'Exposure'
    if 'Exposure' in controls and controls['Exposure']['IsAutoSupported']:
        print('Enabling auto-exposure mode')
        camera.set_control_value(asi.ASI_EXPOSURE,
                                controls['Exposure']['DefaultValue'],
                                auto=True)

    if 'Gain' in controls and controls['Gain']['IsAutoSupported']:
        print('Enabling automatic gain setting')
        camera.set_control_value(asi.ASI_GAIN,
                                controls['Gain']['DefaultValue'],
                                auto=True)

    # Keep max gain to the default but allow exposure to be increased to its maximum value if necessary
    camera.set_control_value(
        controls['AutoExpMaxExpMS']['ControlType'], controls['AutoExpMaxExpMS']['MaxValue'])

    camera.set_image_type(asi.ASI_IMG_RAW8)
    camera.set_roi(width=cam_width, height=cam_height)

    return camera, camera_name


windows_name = ["raw_image", "image_stacking", "image_enhancement", "power_law", "clahe", "histogram_equalization", "fourier",
                "histogram", "blur", "canny_edge", "circle_hough_transform"]

removed = []

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

    def moveWindow(self, x, y):
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
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def buildHistogramFromGrayImage(image):
    h = np.zeros((300, 256))
    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0)]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([image], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
    h = np.flipud(h)
    return h


def saveConfiguration(filename):
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

def isGrayImage(img) :
    return len(img.shape) < 3

def signal_handler(sig, frame):
    saveConfiguration("parameters.txt")
    sys.exit(0)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

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
    windows["circle_hough_transform"].addTrackbar(
        "cht_min_dist", 0, 300, callback)
    windows["circle_hough_transform"].addTrackbar(
        "cht_min_radius", 0, 300, callback)
    windows["circle_hough_transform"].addTrackbar(
        "cht_max_radius", 0, 400, callback)
    windows["circle_hough_transform"].setTrackbarPos(
        "cht_min_dist", cht_min_dist)
    windows["circle_hough_transform"].setTrackbarPos(
        "cht_min_radius", cht_min_radius)
    windows["circle_hough_transform"].setTrackbarPos(
        "cht_max_radius", cht_max_radius)

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
            enhanced_image = raw.copy()

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

            blur_size = windows["blur"].getTrackbarPos("blur_size")
            blur = cv2.GaussianBlur(
                enhanced_image, (blur_size*2 + 1, blur_size*2 + 1), 0)
            windows["blur"].setImage(blur)
            windows["blur"].showWindow()

            if isGrayImage(blur) :
                histogram = buildHistogramFromGrayImage(blur)
            else :
                histogram = buildHistogramFromImage(blur)
            windows["histogram"].setImage(histogram)
            windows["histogram"].showWindow()

            canny_min_val = windows["canny_edge"].getTrackbarPos("canny_min_val")
            canny_max_val = windows["canny_edge"].getTrackbarPos("canny_max_val")
            edge = cv2.Canny(blur, canny_min_val, canny_max_val)
            windows["canny_edge"].setImage(edge)
            windows["canny_edge"].showWindow()

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


            windows["raw_image"].moveWindow(50, 100)
            windows["image_stacking"].moveWindow(475, 100)
            windows["image_enhancement"].moveWindow(900, 100)
            windows["power_law"].moveWindow(1325, 100)
            windows["clahe"].moveWindow(1325, 100)
            windows["histogram_equalization"].moveWindow(1325, 100)
            windows["fourier"].moveWindow(1325, 100)

            windows["histogram"].moveWindow(50, 550)
            windows["blur"].moveWindow(475, 550)
            windows["canny_edge"].moveWindow(900, 550)
            windows["circle_hough_transform"].moveWindow(1325, 550)

            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                break
    except (KeyboardInterrupt, SystemExit):
        camera.close()
        out.release()
    
