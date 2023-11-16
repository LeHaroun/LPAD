import cv2
import numpy as np
from PIL import Image

class PlateDetector:
    def __init__(self):
        self.scale_factor = 0.00392
        self.blob_size = (320, 320)
        self.mean_subtraction = (0, 0, 0)
        self.swapRB = True
        self.crop = False
        self.conf_threshold = 0.3
        self.nms_threshold = 0.1

    def load_model(self, weight_path: str, cfg_path: str):
        self.net = cv2.dnn.readNet(weight_path, cfg_path)
        with open("classes-detection.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()
        self.output_layers = [self.layers_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {img_path}")
        height, width, channels = img.shape
        return img, height, width, channels

    def detect_plates(self, img):
        blob = cv2.dnn.blobFromImage(img, scalefactor=self.scale_factor, size=self.blob_size, mean=self.mean_subtraction, swapRB=self.swapRB, crop=self.crop)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return blob, outputs

    def draw_labels(self, boxes, confidences, class_ids, img):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        plates = []
        for i in indexes:
            try:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                crop_img = self.crop_and_resize(img, x, y, w, h)
                crop_img = self.correct_skew(crop_img)
                plates.append(crop_img)
                self.draw_box_and_text(img, x, y, w, h, confidences[i])
            except Exception as err:
                print(f"Error processing box {i}: {err}")

        return img, plates

    def crop_and_resize(self, img, x, y, w, h):
        crop_img = img[y:y+h, x:x+w]
        return cv2.resize(crop_img, dsize=(470, 110))

    def determine_skew_angle(self, image, delta, limit):
         # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is not None:
            # Calculate the angle of each line
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.rad2deg(theta) - 90
                if abs(angle) <= limit:
                    angles.append(angle)
            if angles:
                # Compute the median angle
                median_angle = np.median(angles)
                return median_angle
        return 0  # Return 0 if no angle is found or lines are not detected

    def correct_skew(self, image):
        angle = self.determine_skew_angle(image, delta=1, limit=5)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


    def draw_box_and_text(self, img, x, y, w, h, confidence):
        font = cv2.FONT_HERSHEY_PLAIN
        color_green = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color_green, 8)
        cv2.putText(img, f"{round(confidence, 3) * 100}%", (x + 20, y - 20), font, 12, color_green, 6)

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
                    center_x, center_y, w, h = self.extract_box_dimensions(detection, width, height)

                    # Validate box dimensions
                    x, y, w, h = self.validate_box_dimensions(center_x, center_y, w, h, width, height)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def extract_box_dimensions(self, detection, width, height):
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        return center_x, center_y, w, h

    def validate_box_dimensions(self, center_x, center_y, w, h, width, height):
        x = max(0, min(center_x - w // 2, width - w))
        y = max(0, min(center_y - h // 2, height - h))
        w = min(w, width - x)
        h = min(h, height - y)
        return x, y, w, h