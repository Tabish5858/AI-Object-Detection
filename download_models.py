import urllib.request
import os
import requests

def download_yolo_files():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # YOLOv4 files to download
    files = {
        'models/yolov4.cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
        'models/coco.names': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
    }

    # Download configuration and class names
    for file_path, url in files.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {file_path}")

    # Download YOLOv4 weights
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    weights_path = "models/yolov4.weights"
    
    if not os.path.exists(weights_path):
        print("Downloading YOLOv4 weights (this may take a while)...")
        response = requests.get(weights_url, stream=True)
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Downloaded YOLOv4 weights")

if __name__ == "__main__":
    download_yolo_files()