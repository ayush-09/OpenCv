# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:00:42 2020

@author: ayush
"""
import cv2
import numpy as np
car_classifier=cv2.CascadeClassifier('A:\OpenCv\cars.xml')
cap=cv2.VideoCapture(r'A:\OpenCv\video.mp4')
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars=car_classifier.detectMultiScale(gray,1.4,2)
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,
                      (x,y),
                      (x+w,y+h),
                      (25,125,225),
                      2)
        cv2.imshow('Car Detection',frame)
    if cv2.waitKey(10) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()