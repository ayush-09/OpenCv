# -*- coding: utf-8 -*-

"""
Created on Sun Aug  9 19:13:04 2020

@author: ayush
"""

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r'A:\OpenCv\Car Plate Detector\car.jpg')
def display(img):
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111)
    new_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
plate_classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_russian_plate_number.xml')
def detect_plate(img):
    plate_img=img.copy()
    gray=cv2.cvtColor(plate_img,cv2.COLOR_BGR2GRAY)
    plate_rects=plate_classifier.detectMultiScale(gray,1.1,1)
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,
                      (x,y),
                      (x+w,y+h),
                      (255,255,255),
                      3)
    return plate_img
result=detect_plate(img)
display(result)
result9 = cv2.resize( 
    result, None, fx = 2, fy = 2,  
    interpolation = cv2.INTER_CUBIC)
result1=cv2.cvtColor(result9,cv2.COLOR_BGR2RGB)
result2=cv2.GaussianBlur(result1,(5,5),0)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
config1=r'--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
new_result = pytesseract.image_to_string(result2,lang='eng',config =config1)
filter_new_predicted_result = "".join(new_result.split()) 
print(filter_new_predicted_result)