# -*- coding: utf-8 -*-

import cv2
import math
print(cv2.__version__)

# classifier file
cascade_src = 'cars.xml'
video_src = 'dataset/v2.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

# scale video feed
s = 1.0

# height of car IRL in meters
dh = 2.0

# camera specs
camdim = [4.89,3.67]    # width and height of CCD in mm
fl = 4.12               # focal length of lense in meters


while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break

    height, width = img.shape[:2]

    img = cv2.resize(img,(int(s*width), int(s*height)), interpolation = cv2.INTER_CUBIC)

    # ratio of height in pixels on screen to distance IRL in meters
    q = (fl * dh * width * s) / camdim[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    k = img
    for (x,y,w,h) in cars:
        cv2.rectangle(k,(x,y),(x+w,y+h),(0,0,255),2)

        cv2.putText(k, "{0:.2f}".format(q / h) + 'm', (int(x+w/2),int(y+h/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0))

    cv2.imshow('video', k)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()