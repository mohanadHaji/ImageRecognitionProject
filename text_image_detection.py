#! /usr/bin/python3
import cv2
import numpy as np

########################################            Image Reading                       ###########################################
image = cv2.imread("/home/mohanad/Desktop/Vision/project/PRImA Layout Analysis Dataset/Images/00000820.tif")
im2 = np.copy(image)

########################################             Pre-Processing Image                ###########################################

#####   Image Posterization ########
im2[im2 >= 160]= 255
im2[im2 < 160] = 0

#####   Grayscale Image ########
gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

#####   Bluring Image  ########
blur = cv2.GaussianBlur(gray, (9,9), 2)

#####   Thresholding Image  ########
thresh = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,1) # this will remove back groung color 
thresh = cv2.bitwise_not(thresh)

#####   Scaling Attributes  ########
width = int(image.shape[1] * 25 / 100)
height = int(image.shape[0] * 25 / 100)
scale = (width, height)
resized = cv2.resize(im2, scale, interpolation=cv2.INTER_AREA)
cv2.imshow('Filtered',resized)

#####   Image Dilation   ########
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,2))
dilate = cv2.dilate(thresh, kernel, iterations=1)
resized = cv2.resize(dilate, scale, interpolation=cv2.INTER_AREA)
cv2.imshow('Image Dilation',resized)

########################################             Seperating Pictures From Text                ###########################################

#####   Creating Black Image   ########
pictures = np.copy(gray) *0
border = np.copy(gray) *0

#####   Finding Big Dilated Areas and Fill it   ########

contours = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours[0]:
    x,y,w,h = cv2.boundingRect(cnt)
    if h >=300 and w>=50 :
        cv2.drawContours(pictures, [cnt], -1, 255, cv2.FILLED)
resized = cv2.resize(pictures, scale, interpolation=cv2.INTER_AREA)
cv2.imshow('Fill Dilated Areas',resized)


pictures = cv2.rectangle(pictures, (0, 0), (pictures.shape[1],pictures.shape[0]), (0,0,0), 50)



#####   Removing Pictures From The Dilated Image   ########
masked_image = cv2.subtract(dilate, pictures)


masked_image = cv2.rectangle(masked_image, (0, 0), (masked_image.shape[1],masked_image.shape[0]), (0,0,0), 50) # remove the boundarys of the image
#####   Fill Picture Dilation with Rectangle around it   ########
cnts = cv2.findContours(pictures, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for c in cnts[0]:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(pictures, (x, y), (x + w, y + h), (255,255,255), cv2.FILLED)

resized = cv2.resize(pictures, scale, interpolation=cv2.INTER_AREA)
cv2.imshow('Picture Mask (Filled)',resized)

#####   Bounding Text Box   ########

contours, hierarchy = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(masked_image,(x,y),(x+w,y+h),255,cv2.FILLED)
resized = cv2.resize(dilate, scale, interpolation=cv2.INTER_AREA)
resized1 = cv2.resize(masked_image, scale, interpolation=cv2.INTER_AREA)
cv2.imshow('Mask Dilation',resized)
cv2.imshow('Masked Image',resized1)

#####   Bounding Text for more accuracy   ########
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
for x in range(1):
    contours, hierarchy = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(masked_image, (x, y), (x + w, y + h), 255, cv2.FILLED)
    masked_image= cv2.dilate(masked_image, kernel, iterations=1)
closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,10))
masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, closing_kernel)
resized1 = cv2.resize(masked_image, scale, interpolation=cv2.INTER_AREA)
cv2.imshow('Bounded Mask',resized1)

########################################             Image & Text Contouring                ###########################################

#####   Image Contouring With Blue Lines   ########
contours = cv2.findContours(masked_image , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image  = cv2.drawContours(image, contours[0], -1, (0,0,255), 5)

#####   Text Contouring With Red Lines   ########
contours = cv2.findContours(pictures , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image, contours[0], -1, (255,0,0), 5)

########################################             Final Result                ###########################################

cv2.imwrite("/home/mohanad/Desktop/Vision/project/PRImA Layout Analysis Dataset/output/00000912.tif", image)
resized = cv2.resize(image, scale, interpolation=cv2.INTER_AREA)
cv2.imshow('image', resized)
cv2.waitKey()
