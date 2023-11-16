import cv2
import numpy as np
import pytesseract
from PIL import Image

class PlateReader:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_PLAIN

    def load_model(self, weight_path: str, cfg_path: str):
        self.net = cv2.dnn.readNet(weight_path, cfg_path)
        with open("classes-ocr.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()

        unconnected_layers = self.net.getUnconnectedOutLayers()
        if unconnected_layers.ndim == 1:  # Scalar indices
            self.output_layers = [self.layers_names[i - 1] for i in unconnected_layers]
        else:  # Assume it's an array of indices
            self.output_layers = [self.layers_names[i[0] - 1] for i in unconnected_layers]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        characters = []
        for i in indexes:
            box = boxes[i]
            x, y, w, h = box
            self.draw_label(img, x, y, w, h, confidences[i], i)
            label = str(self.classes[class_ids[i]])
            characters.append((label, x))
        characters.sort(key=lambda x: x[1])
        plate = self.convert_to_plate_string(characters)
        return img, plate

    def draw_label(self, img, x, y, w, h, confidence, i):
        color = self.colors[i % len(self.colors)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{confidence:.2f}%", (x, y - 6), self.font, 1, color, 2)

    def convert_to_plate_string(self, characters):
        plate = ""
        for label, _ in characters:
            plate += self.convert_to_arabic_if_needed(label)
        return plate

    def convert_to_plate_string(self, characters):
        plate = ""
        for label, _ in characters:
            plate += self.convert_to_arabic_if_needed(label)

        # Handle the specific pattern of numbers followed by 'ww'
        plate = self.handle_ww_pattern(plate)

        return plate

    def handle_ww_pattern(self, plate):
        if 'w' in plate:
            # Extract the number part before 'ww'
            print(plate)
            number_part = plate.split('ww')[0]
            # Remove any spaces or decorative elements
            number_part = ''.join(filter(str.isdigit, number_part))
            # Reconstruct the plate with 'ww'
            return number_part + ' ww'
        return plate

    def convert_to_arabic_if_needed(self, label):
        arabic_mappings = {'أ': 'A', 'ب': 'B', 'ج': 'J', 'د': 'D', 'ه': 'H', 'و': 'W', 'ي': 'Y'}
        return arabic_mappings.get(label, label)

    def tesseract_ocr(self, image, lang="eng", psm=7):
        try:
            alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            options = f"-l {lang} --psm {psm} -c tessedit_char_whitelist={alphanumeric}"
            return pytesseract.image_to_string(image, config=options)
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""