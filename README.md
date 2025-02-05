# AI Object Detection System

## Overview
An advanced computer vision system integrating **YOLO v4** and **MobileNet SSD** models to detect, analyze, and describe objects in images and video streams. The system features a modern and responsive **Flask** web interface for user interactions.

## Key Features
### 1. **Object Detection**
- Multi-model detection using `detector.ObjectDetector`
- Support for 80+ object classes
- High-accuracy detection with confidence scoring
- Real-time processing capabilities

### 2. **Advanced Analysis**
- Color detection and analysis
- Object size estimation
- Position tracking within the frame
- Detailed object descriptions via `descriptions.get_description`

### 3. **Web Interface**
- Modern responsive design
- Real-time visualization of detection results
- Interactive results display
- Support for image and video uploads

## Technical Architecture
### **Backend Components**
- `app.py`: Flask web server and main application logic
- `detector.py`: Core detection engine
- `descriptions.py`: Object description generation
- `utils.py`: Utility functions for image processing

### **Frontend Components**
- `index.html`: Main web interface
- `style.css`: Custom styling
- `main.js`: Client-side interactions

### **Models**
- **YOLO v4**: Primary object detection
- **MobileNet SSD**: Secondary validation
- Models are located in the `models/` directory

## Installation
### **Prerequisites**
- Python ≥ 3.7
- CUDA ≥ 10.2 (for GPU support)
- RAM ≥ 8GB
- Storage ≥ 1GB

### **Setup**
```bash
# Clone the repository
git clone https://github.com/Tabish5858/AI-Object-Detection.git
cd AI-Object-Detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python download_models.py
```

## Configuration
Edit `config.py` to customize the following settings:
- Model paths
- Detection thresholds
- Input image size
- GPU usage

## Usage
### **Starting the Server**
```bash
python app.py
```
Access the web interface at: [http://localhost:5000](http://localhost:5000)

### **API Endpoints**
#### **Image Detection**
- **POST** `/detect`
  - Input: Multipart form with `image` file
  - Output: JSON with detection results

#### **Health Check**
- **GET** `/`
  - Returns: Web interface

## Example Usage
```python
from detector import ObjectDetector

# Initialize detector
detector = ObjectDetector()

# Process image
results = detector.detect(image)

# Access results
for detection in results:
    print(f"Found {detection['class_name']} with {detection['confidence']}% confidence")
```

## Project Structure
```
├── app.py              # Flask application
├── config.py           # Configuration settings
├── detector.py         # Object detection logic
├── descriptions.py     # Object descriptions
├── download_models.py  # Model downloader
├── utils.py            # Utility functions
├── video_detection.py  # Video processing
├── models/             # Model files
│   ├── yolov4.cfg
│   ├── yolov4.weights
│   └── coco.names
├── static/             # Static assets
│   ├── css/
│   └── js/
└── templates/          # HTML templates
```

## Contributing
1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit changes:
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Acknowledgments
- **YOLO v4 Team:** For the primary object detection model
- **MobileNet SSD Developers:** For the secondary validation model
- **Flask Framework Community:** For the powerful web framework

## Support
For issues, please open an issue in the GitHub repository or contact the maintainers.

