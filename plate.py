# -*- coding: utf-8 -*-
"""License_Plate_Detection_OCR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-bDfYGe5oRpOMAdmCu6aJox9cE92BUvD

# License Plate Detection and OCR
This notebook demonstrates a workflow for detecting license plates in images and extracting text from them using YOLO models.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image

# normal CPU TF
import tensorflow as tf

"""
## Model Loading
Here we define classes for plate detection and OCR and load the pre-trained models.
"""

class PlateDetector:
    def load_model(self, weight_path: str, cfg_path: str):
        self.net = cv2.dnn.readNet(weight_path, cfg_path)
        with open("classes-detection.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()
        self.output_layers = [self.layers_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        return img, height, width, channels

    def detect_plates(self, img):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return blob, outputs

    def get_boxes(self, outputs, width, height, threshold=0.3):
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids

    def draw_labels(self, boxes, confidences, class_ids, img):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        font = cv2.FONT_HERSHEY_PLAIN
        plats = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color_green = (0, 255, 0)
                crop_img = img[y:y+h, x:x+w]
                try:
                    crop_resized = cv2.resize(crop_img, dsize=(470, 110))
                    plats.append(crop_resized)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color_green, 8)
                    confidence = round(confidences[i], 3) * 100
                    cv2.putText(img, str(confidence) + "%", (x + 20, y - 20), font, 12, (0, 255, 0), 6)
                except cv2.error as err:
                    print(err)

        return img,plats

class PlateReader:
    def load_model(self, weight_path: str, cfg_path: str):
        self.net = cv2.dnn.readNet(weight_path, cfg_path)
        with open("classes-ocr.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()
        self.output_layers = [self.layers_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        return img, height, width, channels

    def read_plate(self, img):
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return blob, outputs

    def get_boxes(self, outputs, width, height, threshold=0.3):
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def draw_labels(self, boxes, confidences, class_ids, img):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        font = cv2.FONT_HERSHEY_PLAIN
        c = 0
        characters = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i % len(self.colors)]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
                confidence = round(confidences[i], 3) * 100
                cv2.putText(img, str(confidence) + "%", (x, y - 6), font, 1, color, 2)
                characters.append((label, x))
        characters.sort(key=lambda x:x[1])
        plate = ""
        for l in characters:
            plate += l[0]
        chg = 0
        for i in range(len(plate)):
            if plate[i] in ['b', 'h', 'd', 'a']:
                if plate[i-1] == 'w':
                    ar = i-1
                    chg = 2
                elif plate[i-1] == 'c':
                    ar = i - 1
                    chg = 3
                else:
                    ar = i
                    chg = 1

        if chg == 1:
            plate = plate[:ar] + str(self.arabic_chars(ord(plate[ar])), encoding="utf-8") + plate[ar+1:]
        if chg == 2:
            index = 0
            for i in range(3):
                index = index + plate[ar+i]
            plate = plate[:ar] + str(self.arabic_chars(index), encoding="utf-8")  + plate[ar+3:]
        if chg == 3:
            index = 0
            for i in range(2):
                index = index + plate[ar+i]
            plate = plate[:ar] + str(self.arabic_chars(index), encoding="utf-8") + plate[ar+2:]

        return img, plate

    def arabic_chars(self, index):
        if (index == ord('a')):
            return "أ".encode("utf-8")

        if (index == ord('b')):
            return "ب".encode("utf-8")

        if (index == 2 * ord('w') + ord('a') or index == ord('w')):
            return "و".encode("utf-8")

        if (index == ord('d')):
            return "د".encode("utf-8")

        if (index == ord('h')):
            return "ه".encode("utf-8")

        if (index == ord('c') + ord('h')):
            return "ش".encode("utf-8")

    def tesseract_ocr(self, image, lang="eng", psm=7):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-l {} --psm {} -c tessedit_char_whitelist={}".format(lang, psm, alphanumeric)
        return pytesseract.image_to_string(image, config=options)
        #reshape ???

# Load models, check conflicts with opencv !
detector = PlateDetector()
detector.load_model('detection/yolov3-detection_final.weights', 'detection/yolov3-detection.cfg')

reader = PlateReader()
reader.load_model('ocr/yolov3-ocr_final.weights', 'ocr/yolov3-ocr.cfg')

"""
## Integrated Processing Pipeline
This function integrates image loading, detection, and OCR.
"""

def process_image(image_path):
    # Load image
    img, height, width, channels = detector.load_image(image_path)

    # Detect plates
    blob, outputs = detector.detect_plates(img)
    boxes, confidences, class_ids = detector.get_boxes(outputs, width, height, threshold=0.3)
    plate_img, LpImg = detector.draw_labels(boxes, confidences, class_ids, img)

    # Check if any plates were detected
    if not LpImg:
        return "No plates detected, Try to change Camera angle or Lighting Conditions"

    # Perform OCR on the first detected plate
    ocr_image = LpImg[0]  # assuming the first detected plate
    blob, outputs = reader.read_plate(ocr_image)
    boxes, confidences, class_ids = reader.get_boxes(outputs, width, height, threshold=0.3)
    segmented, plate_text = reader.draw_labels(boxes, confidences, class_ids, ocr_image)

    # Return OCR result
    return plate_text


# Usage usage
print("Normal cases")
print(process_image('images/Example1.jpg'))
print(process_image('images/Example2.jpg'))
print(process_image('images/Example3.jpg'))
print("Edge cases 1")
print(process_image('images/Example4.jpg'))
print(process_image('images/Example5.jpg'))
print("Edge cases 2")
#print(process_image('images/Example6.jpg'))
#print(process_image('images/Example7.jpg'))
print(process_image('images/Example8.jpg'))



