from matplotlib import pyplot as plt
import argparse
import os
import sys
import time
import zwoasi as asi

import signal
import cv2
import numpy as np

def callback(x):
    pass

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


def clahe(image, clipLimit=2.0, tileGridSize=8):
    if not isGrayImage(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (tileGridSize == 0):
        tileGridSize = 1
    clahe = cv2.createCLAHE(clipLimit, (tileGridSize, tileGridSize))
    image = clahe.apply(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.where(image > 254, 254, image)
    image = np.where(image < 0, 0, image)
    image = image.astype(np.uint8)
    return image


def fourierTransform(image):
    if not isGrayImage(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = image.shape
    crow, ccol = int(rows/2), int(cols/2)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    image_back = cv2.idft(f_ishift)
    image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

    # Convert back again
    # normalize the data to 0 - 1
    image_back = image_back.astype(np.float32) / image_back.max()
    image_back = 255 * image_back  # Now scale by 255
    image = image_back.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


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


def isGrayImage(img):
    return len(img.shape) < 3


def buildHistogramFromImage(image):
    h = None
    color = None
    if isGrayImage(image):
        h = np.zeros((300, 256))
        color = [(255, 0, 0)]
    else:
        h = np.zeros((300, 256, 3))
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    bins = np.arange(256).reshape(256, 1)
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([image], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
    h = np.flipud(h)
    return h

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
    camera.set_control_value(asi.ASI_EXPOSURE, 49)
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
    # k = 'Exposure'
    # if 'Exposure' in controls and controls['Exposure']['IsAutoSupported']:
    #     print('Enabling auto-exposure mode')
    #     camera.set_control_value(asi.ASI_EXPOSURE,
    #                             controls['Exposure']['DefaultValue'],
    #                             auto=True)

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
