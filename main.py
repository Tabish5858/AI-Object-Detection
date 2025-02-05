import cv2
from detector import ObjectDetector
from utils import draw_predictions
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", 
                        help="Video source (0 for webcam)")
    args = parser.parse_args()
    
    detector = ObjectDetector()
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        detections, fps = detector.detect(frame)
        frame = draw_predictions(frame, detections, fps, detector.config.CLASSES)
        
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()