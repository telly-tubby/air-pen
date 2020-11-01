import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)
canvas = None
x1, y1 = 0,0
noiseth = 500
kernel = np.ones((5, 5), np.uint8)
while True:
    ret, img = cap.read()
    img = cv2.flip( img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lr = np.array([84, 188, 88])
    ur = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lr, ur)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea))>noiseth:
        c = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if x1 == 0 and y1 == 0:
            x1, y1 = x, y
        else:
            canvas = cv2.line(canvas,(x1,y1),(x,y),[0,0,255],3)
            x1, y1 = x, y
    else:
        x1, y1 = 0,0

    img = cv2.add(img, canvas)
    cv2.imshow('frame', img)
    cv2.imshow('canvas', canvas)




    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()