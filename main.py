# -*- coding: utf-8 -*-

import cv2
import math
print(cv2.__version__)

cascade_src = 'cars.xml'
video_src = 'dataset/v2.mp4'
#video_src = 'dataset/video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
reti, imgo = cap.read()
s = 1.0


dh = 2.0
camdim = [4.89,3.67]
fl = 4.12

#fgbg = cv2.createBackgroundSubtractorKNN(500,400)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break

    height, width = img.shape[:2]

    img = cv2.resize(img,(int(s*width), int(s*height)), interpolation = cv2.INTER_CUBIC)


    #FoV = 2*math.atan((camdim[1]/2)/fl)
    #flpx = (height * 0.5) / math.tan(FoV * 0.5)

    q = (fl * dh * width * s) / camdim[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = fgbg.apply(img) #cv2.absdiff(img,imgo)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    k = img
    for (x,y,w,h) in cars:
        cv2.rectangle(k,(x,y),(x+w,y+h),(0,0,255),2)

        cv2.putText(k, "{0:.2f}".format(q / h) + 'm', (int(x+w/2),int(y+h/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0))

    cv2.imshow('video', k)

    if cv2.waitKey(33) == 27:
        break

    imgo = img
cv2.destroyAllWindows()