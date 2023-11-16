import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
import re

from plate_detector import PlateDetector
from plate_reader import PlateReader

class MainApplication(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)

        # Initialize the video_frame attribute
        self.video_frame = None

        # Initialize plate detector and reader
        self.detector = PlateDetector()
        self.detector.load_model('weights/yolov3-detection_final.weights', 'weights/yolov3-detection.cfg')
        self.reader = PlateReader()
        self.reader.load_model('weights/yolov3-ocr_final.weights', 'weights/yolov3-ocr.cfg')

        # UI Elements
        self.setup_ui()

        # Video capture setup
        self.cap = cv2.VideoCapture('video/Example1.MP4')
        self.frame_extract_rate = 30  # Extract and process every 'x' frames
        self.frame_counter = 0

        # Control Variables
        self.is_playing = False
        self.after_id = None

        self.detected_plates = [] 


    def setup_ui(self):
        self.create_video_frame()

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

    def create_video_frame(self):
        if self.video_frame is not None:
            self.video_frame.destroy()
        self.video_frame = tk.LabelFrame(self, text="Live Plate Detector Demo")
        self.video_frame.pack(fill="both", expand=True)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill="both")

    def start_video(self):
        if self.is_playing:
            return
        self.is_playing = True
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"
        self.update_video_frame()

    def update_video_frame(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.frame_counter += 1
                if self.frame_counter >= self.frame_extract_rate:
                    self.frame_counter = 0
                    self.process_current_frame(frame)

                # Display the frame
                frame = cv2.resize(frame, (800, 600))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame)
                frame_photo = ImageTk.PhotoImage(image=frame_image)
                self.video_label.imgtk = frame_photo
                self.video_label.configure(image=frame_photo)

            self.after_id = self.master.after(30, self.update_video_frame)

    def process_current_frame(self, frame):
        # Save the current frame to a temporary file
        temp_filename = "temp_frame.jpg"
        cv2.imwrite(temp_filename, frame)

        # Process the image and display results
        plate_text = self.process_image(temp_filename)
        pattern = r'\b\d{4,}[A-Za-z]\d{0,2}\b'
        if "No plates detected" not in plate_text:
            matched_plates = re.findall(pattern, plate_text)
            for plate in matched_plates:
                if plate not in self.detected_plates:
                    self.detected_plates.append(plate)
                    self.output_text.insert(tk.END, plate + "\n")


        # Clean up the temporary file
        os.remove(temp_filename)

    def process_image(self, image_path):
        img, height, width, channels = self.detector.load_image(image_path)
        blob, outputs = self.detector.detect_plates(img)
        boxes, confidences, class_ids = self.detector.get_boxes(outputs, width, height)
        plate_img, LpImg = self.detector.draw_labels(boxes, confidences, class_ids, img)

        if not LpImg:
            return "No plates detected. Try to change camera angle or lighting conditions."

        ocr_image = LpImg[0]
        blob, outputs = self.reader.read_plate(ocr_image)
        boxes, confidences, class_ids = self.reader.get_boxes(outputs, width, height)
        segmented, plate_text = self.reader.draw_labels(boxes, confidences, class_ids, ocr_image)
        return plate_text

    def stop_video(self):
        self.is_playing = False
        if self.after_id:
            self.master.after_cancel(self.after_id)
        if self.cap.isOpened():
            self.cap.release()  # Release the video capture
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"
        self.clear_output()

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    root.title("Plate Detection and Reader")
    app = MainApplication(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()
