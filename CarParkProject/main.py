import cv2
import pickle
import cvzone
import numpy as np

# Video
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 45


def checkParkingSpace(img_Process):
    spaceCounter = 0
    indexCounter = 0

    for pos in posList:
        x, y = pos
        imgCrop = img_Process[y:y + height, x:x + width]
        # cv2.imshow(str(x*y),imgCrop)
        count = cv2.countNonZero(imgCrop)
        # cvzone.putTextRect(img,str(count),(x,y+height-3),scale=1.5, thickness=2, offset=0, colorR=(0,0,255))

        if count < 900:
            color = (0, 255, 0)
            thickness = 4
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        indexCounter += 1
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(indexCounter), (x, y + height - 3), scale=1.5, thickness=2, offset=0,
                           colorR=(0, 0, 0))
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=2, offset=20,
                       colorR=(0, 200, 0))
    # print(spaceCounter)


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)

    kernel = np.ones((3, 3), np.uint8) / 9
    imgMean = cv2.filter2D(imgMedian, -1, kernel)

    imgSum = imgMedian + imgMean

    imgDilate = cv2.dilate(imgSum, kernel, iterations=1)

    checkParkingSpace(imgDilate)

    cv2.imshow("Parking", img)
    # cv2.imwrite('Parking_image.png', img)
    cv2.imshow("ImageBlur", imgBlur)
    # cv2.imwrite('ImageBlur_image.png', imgBlur)
    cv2.imshow("ImageThresh", imgThreshold)
    #cv2.imwrite('ImageThresh_image.png', imgThreshold)
    cv2.imshow("ImageMedian", imgMedian)
    #cv2.imwrite('ImageMedian_image.png', imgMedian)
    cv2.imshow("ImageMean", imgMean)
    #cv2.imwrite('ImageMean_image.png', imgMean)
    cv2.imshow("ImageSum", imgSum)
    cv2.waitKey(10)
