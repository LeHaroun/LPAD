import tkinter as tk
from tkinter import ttk
from tkvideo import tkvideo
import threading
import cv2
from PIL import Image, ImageTk
import os

from plate_detector import PlateDetector
from plate_reader import PlateReader

class MainApplication(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)

        # Initialize plate detector and reader
        self.detector = PlateDetector()
        self.detector.load_model('weights/yolov3-detection_final.weights', 'weights/yolov3-detection.cfg')
        self.reader = PlateReader()
        self.reader.load_model('weights/yolov3-ocr_final.weights', 'weights/yolov3-ocr.cfg')

        # UI Elements
        self.setup_ui()
        
        # Video Player
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill="both")
        self.player = tkvideo("video/Example1.MP4", self.video_label, loop=1, size=(800, 600))

        # Control Variables
        self.is_playing = False
        self.frame_extraction_rate = 30  # Extract a frame every 'x' frames

    def setup_ui(self):
        self.video_frame = tk.LabelFrame(self, text="Live Plate Detector Demo")
        self.video_frame.pack(fill="both", expand=True)

        self.control_frame = tk.Frame(self)
        self.control_frame.pack(fill="x", expand=False)

        self.start_button = tk.Button(self.control_frame, text="Start", command=self.start_video)
        self.start_button.pack(side="left")

        self.stop_button = tk.Button(self.control_frame, text="Stop", command=self.stop_video, state="disabled")
        self.stop_button.pack(side="left")

        self.clear_output_button = tk.Button(self.control_frame, text="Clear Output", command=self.clear_output)
        self.clear_output_button.pack(side="left")

        self.output_text = tk.Text(self, height=10)
        self.output_text.pack(fill="both", expand=True)

    def start_video(self):
        self.is_playing = True
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"
        self.player.play()
        self.update_video_frame()

    def update_video_frame(self):
        if self.is_playing:
            frame = self.player.get_frame()  # Get the current frame
            if frame is not None:
                self.process_current_frame(frame)
            self.master.after(30, self.update_video_frame)  # 30 milliseconds

    def process_current_frame(self, frame):
        # Save the current frame to a temporary file
        temp_image_path = 'temp_frame.jpg'
        frame.save(temp_image_path)
        
        # Process the frame for plate detection
        plate_text = self.process_image(temp_image_path)
        self.output_text.insert(tk.END, plate_text + "\n")
        
        # Remove the temporary file
        os.remove(temp_image_path)

    def process_image(self, image_path):

        img, height, width, channels = self.detector.load_image(image_path)

        # Detect plates
        blob, outputs = self.detector.detect_plates(img)
        boxes, confidences, class_ids = self.detector.get_boxes(outputs, width, height, threshold=0.3)
        plate_img, LpImg = self.detector.draw_labels(boxes, confidences, class_ids, img)

        # Check if any plates were detected
        if not LpImg:
            return "No plates detected. Try to change Camera angle or Lighting Conditions"

        # Perform OCR on the first detected plate
        ocr_image = LpImg[0]  # assuming the first detected plate
        blob, outputs = self.reader.read_plate(ocr_image)
        boxes, confidences, class_ids = self.reader.get_boxes(outputs, width, height, threshold=0.3)
        segmented, plate_text = self.reader.draw_labels(boxes, confidences, class_ids, ocr_image)

        # Return OCR result
        return plate_text

    def stop_video(self):
        self.is_playing = False
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    root.title("Plate Detection and Reader")
    app = MainApplication(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()
