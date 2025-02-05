from pathlib import Path

BASE_DIR = Path(__file__).parent

class DetectorConfig:
    MODEL_PATH = str(BASE_DIR / "models" / "yolov4.weights")
    CONFIG_PATH = str(BASE_DIR / "models" / "yolov4.cfg")
    NAMES_PATH = str(BASE_DIR / "models" / "coco.names")
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    INPUT_SIZE = (416, 416)
    USE_GPU = False

    def __init__(self):
        # Load COCO class names
        with open(self.NAMES_PATH) as f:
            self.CLASSES = [line.strip() for line in f.readlines()]