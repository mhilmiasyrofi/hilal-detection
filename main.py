import cv2
import numpy as np

windows_name = ["raw_image", "clahe", "power_law", "gamma_transformation",
                "blur", "canny_edge", "circle_hough_transform"]
# LHE = Local Histogram equalization
# removed = ["clahe"]
# removed = ["blur", "canny_edge", "circle_hough_transform"]
removed = []

 

MODE_POWER_LOW = 1
MODE_GAMMA_CORRECTION = 2
MODE_CLAHE = 3
MODE_FOURIER_TRANSFORM = 4

empty_image = np.zeros((1, 1, 3), np.uint8)

for r in removed:
    windows_name.remove(r)


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


def powerLawTransformation(img, power=60):
    power_law = img/255
    power_law = cv2.pow(power_law, power/100)
    power_law = convertFloat64ImgtoUint8(power_law)
    return power_law

def clahe(img, clipLimit=2.0, tileGridSize=8):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (tileGridSize == 0) :
        tileGridSize = 1
    clahe = cv2.createCLAHE(clipLimit, (tileGridSize, tileGridSize))
    img = clahe.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def gammaCorrection(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values

    if gamma < 0.1 :
        gamma = 0.1 
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def fourierTransform(img, magnitude):
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = magnitude * \
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
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print(img.shape)

    return img


if __name__ == "__main__":

    windows = {}

    for w in windows_name:
        windows[w] = Window(w)

    im_name = "data/hilal2.jpg"
    # im_name = "data/hilal1.jpg"
    # im_name = "data/hilal.jpg"
    # im_name = "data/frame.jpg"
    img = cv2.imread(im_name)
    img = resizeImage(img)

    # RAW IMAGE
    image_enhancement_mode = 0
    windows["raw_image"].addTrackbar("image_enhancement_mode", 0, 3, callback)
    windows["raw_image"].setTrackbarPos("image_enhancement_mode", image_enhancement_mode)

    # IMAGE ENHANCEMENT
    power = 60
    windows["power_law"].addTrackbar("power", 0, 100, callback)
    windows["power_law"].setTrackbarPos("power", power)
    
    gamma = 100
    windows["gamma_transformation"].addTrackbar("gamma", 0, 250, callback)
    windows["gamma_transformation"].setTrackbarPos("gamma", gamma)
    clip_limit = 2
    tile_grid_size = 5
    windows["clahe"].addTrackbar("clip_limit", 0, 10, callback)
    windows["clahe"].setTrackbarPos("clip_limit", clip_limit)
    windows["clahe"].addTrackbar("tile_grid_size", 0, 10, callback)
    windows["clahe"].setTrackbarPos("tile_grid_size", tile_grid_size)

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
        enhanced_image = img.copy()

        image_enhancement_mode = windows["raw_image"].getTrackbarPos("image_enhancement_mode")
        if (image_enhancement_mode == MODE_POWER_LOW):
            power = windows["power_law"].getTrackbarPos("power")
            enhanced_image = powerLawTransformation(enhanced_image, power)
            windows["power_law"].setImage(enhanced_image)
            windows["power_law"].showWindow()
            windows["gamma_transformation"].setImage(empty_image)
            windows["gamma_transformation"].showWindow()
            windows["clahe"].setImage(empty_image)
            windows["clahe"].showWindow()
        elif (image_enhancement_mode == MODE_GAMMA_CORRECTION):
            gamma = windows["gamma_transformation"].getTrackbarPos("gamma")
            gamma = gamma/100
            enhanced_image = gammaCorrection(enhanced_image, gamma)
            windows["gamma_transformation"].setImage(enhanced_image)
            windows["gamma_transformation"].showWindow()
            windows["power_law"].setImage(empty_image)
            windows["power_law"].showWindow()
            windows["clahe"].setImage(empty_image)
            windows["clahe"].showWindow()
        elif (image_enhancement_mode == MODE_CLAHE) :
            clip_limit = windows["clahe"].getTrackbarPos("clip_limit")
            tile_grid_size = windows["clahe"].getTrackbarPos("tile_grid_size")
            enhanced_image = clahe(img, clip_limit, tile_grid_size)
            windows["clahe"].setImage(enhanced_image)
            windows["clahe"].showWindow()
            windows["gamma_transformation"].setImage(empty_image)
            windows["gamma_transformation"].showWindow()
            windows["power_law"].setImage(empty_image)
            windows["power_law"].showWindow()
        elif (image_enhancement_mode == MODE_FOURIER_TRANSFORM):
            magnitude = windows["power_law"].getTrackbarPos("power")
            enhanced_image = fourierTransform(enhanced_image, magnitude)
        else :
            windows["gamma_transformation"].setImage(empty_image)
            windows["gamma_transformation"].showWindow()
            windows["power_law"].setImage(empty_image)
            windows["power_law"].showWindow()
            windows["clahe"].setImage(empty_image)
            windows["clahe"].showWindow()


        blur_size = windows["blur"].getTrackbarPos("blur_size")
        blur = cv2.GaussianBlur(enhanced_image, (blur_size*2 + 1, blur_size*2 + 1), 0)
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
                # draw the center of the circle
                cv2.circle(circle_img,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(circle_img,(i[0],i[1]),2,(0,0,255),3)

        windows["circle_hough_transform"].setImage(circle_img)
        windows["circle_hough_transform"].showWindow()

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
