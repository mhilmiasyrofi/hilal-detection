import cv2
import numpy as np
 
img = cv2.imread('data/hilal1.jpg')
h = np.zeros((300,256,3))
 
bins = np.arange(256).reshape(256,1)
color = [ (255,0,0),(0,255,0),(0,0,255) ]
for ch, col in enumerate(color):
    hist_item = cv2.calcHist([img],[ch],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    pts = np.column_stack((bins,hist))
    cv2.polylines(h,[pts],False,col)
 
h=np.flipud(h)
 
cv2.imshow('colorhist',h)

k = cv2.waitKey(1) & 0xFF
if k == 27:
    exit()


if __name__ == "__main__":
    im_name = "data/hilal2.jpg"
    img = cv2.imread(im_name)
