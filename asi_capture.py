#!/usr/bin/env python

import argparse
import os
import sys
import time
import zwoasi as asi

import cv2
import numpy as np
import threading


__author__ = 'Steve Marple'
__version__ = '0.0.22'
__license__ = 'MIT'

global ExpTime
global CamGain
global tframe
global tresh
ExpTime = 100
CamGain = 50
tframe = np.zeros((480, 640, 3))
tresh = np.zeros((480, 640, 1))

def saveControlValues(filename, settings):
    filename += '.txt'
    with open(filename, 'w') as f:
        for k in sorted(settings.keys()):
            f.write('%s: %s\n' % (k, str(settings[k])))
    print('Camera settings saved to %s' % filename)

def gray(im):
    im = 255 * (im/im.max())
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

env_filename = os.getenv('ZWO_ASI_LIB')

parser = argparse.ArgumentParser(
    description='Process and save images from a camera')
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
print('')
print('Camera controls:')
controls = camera.get_controls()
for cn in sorted(controls.keys()):
    print('    %s:' % cn)
    for k in sorted(controls[cn].keys()):
        print('        %s: %s' % (k, repr(controls[cn][k])))


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

class getFrame(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stopThread = False

    def run(self):
        while True:
            global tframe
            global ExpTime
            camera.set_control_value(asi.ASI_EXPOSURE, ExpTime)
            camera.set_control_value(asi.ASI_GAIN, CamGain)
            tframe = camera.capture_video_frame()
            if self.stopThread == True:
                break

    def stopThread(self, stopThread):
        self.stopThread = stopThread


print('Enabling stills mode')
try:
    # Force any single exposure to be halted
    camera.stop_video_capture()
    camera.stop_exposure()
except (KeyboardInterrupt, SystemExit):
    raise
except:
    pass

### warm up camera
time.sleep(1)
i = 0
while (i <= 5) :
    tframe  = camera.capture()
    i += 1

# Enable video mode
try:
    # Force any single exposure to be halted
    camera.stop_exposure()
except (KeyboardInterrupt, SystemExit):
    raise
except:
    pass

print('Enabling video mode')
camera.start_video_capture()


### thread to capture camera
t1 = getFrame()
t1.start()

try :
    while True:
        frame = gray(tframe)
        cv2.imshow("frame", tframe)
except (KeyboardInterrupt, SystemExit) :
    cv2.destroyAllWindows()
