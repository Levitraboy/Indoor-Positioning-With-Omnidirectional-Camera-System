import imutils as imutils
import numpy as np
import cv2


def color_detection(frame):
    liste = []
    array = []
    hsvFrame = np.array(frame)
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    w = 0
    h = 0
    max_x = 0
    min_x = 10000
    max_y = 0
    min_y = 10000
    # Reading the video from the
    # webcam in image frames

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space

    # Set range for blue color and
    # define mask

    blue_lower = np.array([102,121 ,75 ])
    blue_upper = np.array([130, 255,255 ])

    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((3, 3), "uint8")

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        if (area) > 100:
            # using drawContours() function
            n = approx.ravel()
            for k in range(len(n)):
                if k%2 == 0:
                    if abs(n[0] - n[k]) > w:
                        w = abs(n[0] - n[k])

                    if abs(n[1] - n[k+1]) > h:
                        h = abs(n[1] - n[k+1])

                    if n[k] > max_x:
                        max_x = n[k]
                    if n[k] < min_x:
                        min_x = n[k]
                else:
                    if n[k] > max_y:
                        max_y = n[k]
                    if n[k] < min_y:
                        min_y = n[k]
                
                        
            liste.append([min_x, min_y, w, h])

            i = 0
            for j in n:
                if i % 2 == 0:
                    x = n[i]
                    y = n[i+1]

                    string = str(x) + " " + str(y)

                    cv2.putText(hsvFrame, string, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                i = i + 1

        max_x = 0
        min_x = 10000
        max_y = 0
        min_y = 10000
    for i in range(len(liste)):
        if i%2 == 0:
            array.append(liste[i])
    array.sort()
    return array

