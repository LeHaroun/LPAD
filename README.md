# License Plate Detection and OCR Project

## Overview
This project involves a system for detecting license plates in images and extracting text from them using YOLO models and OCR technology.

## Installation
To install the required dependencies for this project, run:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to start the plate detection and OCR process. The system can be used to detect license plates from images and extract the text from them.

```bash
python plate.py
```

## Components
- Plate Detection: Detects license plates in an image using YOLO models.
- OCR: Extracts text from the detected license plate.

## TODO
- Gradio Interface: Simple UI
- Edge cases for ww cars, latin cars
- Edge cases for empty images, skewed
- Properly trained model on UM6P dataset
- Vehicle detection and BB detection 
  
## Contributing
Contributions to the project are welcome. Please ensure to follow the coding standards and submit pull requests for any new features or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
