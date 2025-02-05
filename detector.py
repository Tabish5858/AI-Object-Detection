import cv2
import numpy as np
import logging
from config import DetectorConfig

class ObjectDetector:
    def __init__(self):
        self.config = DetectorConfig()
        self.setup_logging()
        self.load_model()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_dominant_color(self, frame, box):
        """Extract dominant color from detected object region"""
        x1, y1, x2, y2 = [int(b) for b in box]
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "unknown"
            
        # Calculate average color
        avg_color = np.mean(roi, axis=(0, 1))
        # Convert BGR to RGB
        rgb = avg_color[::-1]
        
        # Define color ranges
        colors = {
            "red": ([150, 0, 0], [255, 50, 50]),
            "green": ([0, 150, 0], [50, 255, 50]),
            "blue": ([0, 0, 150], [50, 50, 255]),
            "white": ([200, 200, 200], [255, 255, 255]),
            "black": ([0, 0, 0], [50, 50, 50]),
            "gray": ([70, 70, 70], [170, 170, 170]),
            "brown": ([40, 20, 0], [140, 90, 50]),
            "yellow": ([200, 200, 0], [255, 255, 50])
        }
        
        for color_name, (lower, upper) in colors.items():
            if all(l <= v <= u for v, l, u in zip(rgb, lower, upper)):
                return color_name
                
        return "unknown"

    def get_object_size(self, box, frame_size):
        """Calculate relative size of detected object"""
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        box_area = box_width * box_height
        frame_area = frame_size[0] * frame_size[1]
        size_ratio = box_area / frame_area
        
        if size_ratio > 0.5:
            return "very large"
        elif size_ratio > 0.25:
            return "large"
        elif size_ratio > 0.1:
            return "medium"
        else:
            return "small"

    def get_position(self, box, frame_size):
        """Determine object position in frame"""
        center_x = (box[0] + box[2]) / 2 / frame_size[0]
        center_y = (box[1] + box[3]) / 2 / frame_size[1]
        
        if center_y < 0.33:
            v_pos = "top"
        elif center_y < 0.66:
            v_pos = "middle"
        else:
            v_pos = "bottom"
            
        if center_x < 0.33:
            h_pos = "left"
        elif center_x < 0.66:
            h_pos = "center"
        else:
            h_pos = "right"
            
        return f"{v_pos} {h_pos}"

    def load_model(self):
        """Load YOLO model"""
        try:
            self.net = cv2.dnn.readNetFromDarknet(
                self.config.CONFIG_PATH,
                self.config.MODEL_PATH
            )
            
            if self.config.USE_GPU:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame):
        """Detect objects using YOLO"""
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            self.config.INPUT_SIZE, 
            swapRB=True, 
            crop=False
        )
        
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return self.process_detections(outputs, frame)

    def detect(self, frame):
        """Detect objects using YOLO"""
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            self.config.INPUT_SIZE, 
            swapRB=True, 
            crop=False
        )
        
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return self.process_detections(outputs, frame)

    def process_detections(self, outputs, frame):
        """Process YOLO detections"""
        results = []
        (H, W) = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        # Process each output layer
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.config.CONFIDENCE_THRESHOLD:
                    # YOLO returns center coordinates and dimensions
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Calculate corner coordinates
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        idxs = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.config.CONFIDENCE_THRESHOLD, 
            self.config.NMS_THRESHOLD
        )

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                color = self.get_dominant_color(frame, [x, y, x + w, y + h])
                size = self.get_object_size([x, y, x + w, y + h], (W, H))
                
                results.append({
                    'class_id': class_ids[i],
                    'class_name': self.config.CLASSES[class_ids[i]],
                    'confidence': confidences[i],
                    'box': np.array([x, y, x + w, y + h]),
                    'color': color,
                    'size': size,
                    'position': self.get_position([x, y, x + w, y + h], (W, H))
                })

        return results