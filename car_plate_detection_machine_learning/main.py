import cv2
import numpy as np
import glob
import os.path
from PIL import Image
import pytesseract
import random


def OCR(plate):
    h = plate.shape[0]
    scale = 40/h
    plate = cv2.resize(plate, (0, 0), fx=scale, fy=scale)
    filtered = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    filtered = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Threshold Plate2", filtered)
    mask = np.zeros(filtered.shape, dtype=np.uint8)
    cnts = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 15:
            cv2.fillPoly(filtered, [c], (0, 0, 0))
        if area > 250:
            cv2.fillPoly(filtered, [c], (0, 0, 0))

    kernel = np.ones((2, 1), np.uint8)
    filtered = dilation = cv2.erode(filtered, kernel, iterations=1)
    coords = np.column_stack(np.where(filtered > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = filtered.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(filtered, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.imshow("Threshold Plate", rotated)
    text = pytesseract.image_to_string(rotated, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789AH')
    print(text)
    return text

#############################    LOADING IMAGES     #############################
images_path = glob.glob(r"~/Desktop/vision/p2/PalestinePlateDataSet/images/*.jpg")
with open('skip_Images.txt') as f:
    skip_Images = [line.rstrip() + ".jpg" for line in f]
anno_path = glob.glob(r"~/Desktop/vision/p2/PalestinePlateDataSet/annotation/*.txt")
OutputPath = '~/Desktop/vision/p2/PalestinePlateDataSet/Output'

#############################    LOADING NEURAL NETWORK CLASSIFIER     #############################
weightsPath = "yolov3_training.weights"
configPath = "yolov3.cfg"
DNN = cv2.dnn.readNet(weightsPath, configPath)
YOLO_Layer = DNN.getLayerNames()
YOLO_Layer = [YOLO_Layer[i[0] - 1] for i in DNN.getUnconnectedOutLayers()]
classes = ["plate"]

##  Loop To Classify All Images  ###################
for im_pth in images_path:
    if os.path.basename(im_pth) not in skip_Images:
        img = cv2.imread(im_pth)

        height, width = img.shape[:2]

    ##  Creating Blob Image And Computer Output Of Layers ###################
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        DNN.setInput(blob)
        FeatureMap = DNN.forward(YOLO_Layer)

    ##  Deep Neural Network Object Detection ###################
        class_ids = []
        HitRate = []
        annotation = []
        for out in FeatureMap:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                hitRate = scores[class_id]
                if hitRate > 0.5:
                    print(hitRate)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    annotation.append([x, y, w, h])
                    HitRate.append(float(hitRate))
                    class_ids.append(class_id)

    ##  Applying  NonMaxSuppression ###################
        NMS = cv2.dnn.NMSBoxes(annotation, HitRate, 0.5, 0.4)
        for i in range(len(annotation)):
            if i in NMS:
                x, y, w, h = annotation[i]
                label = str(classes[class_ids[i]]) + "/"+ str(HitRate[i])
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)
                cv2.putText(img, label, (x, y +2*h), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            plate = img[y:y + h, x:x + w]
            cv2.imshow("Plate Detection", img)
            cv2.imshow("Cropped Plate", plate)
            plateNumber = OCR(plate)

        cv2.waitKey(0)
