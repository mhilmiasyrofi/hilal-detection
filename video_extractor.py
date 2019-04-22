import cv2
import numpy as np

def convertFloat64ImgtoUint8(img):
    # Get the information of the incoming image type
    # normalize the img to 0 - 1
    img = img.astype(np.float64) / float(img.max())
    img = 255 * img  # Now scale by 255
    img = img.astype(np.uint8)
    return img

if __name__ == "__main__":

    # Create a VideoCapture object
    filename = "data/Video Hilal/Data1/Flat/10_39_12.avi"
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
  
    i = 0

    sum_images = None
    while(True):
        ret, frame = cap.read()

        if ret == True:
            # Display the resulting frame
            cv2.imshow('frame', frame)
            # print (type(frame))
            print(i)
            if i == 0:
                # print(frame)
                sum_images = frame.astype(np.float64)
            else:
                sum_images += frame
            i += 1

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    normalized = sum_images/i
    normalized_image = normalized.astype(np.uint8)
    
    # cv2.imwrite('frame.jpg', normalized_image)
