import cv2
import numpy as np
import time

def calculate_fps(start_time):
    return 1.0 / (time.time() - start_time)

def draw_predictions(frame, detections, fps, classes):
    for detection in detections:
        box = detection['box']
        label = f"{classes[detection['class_id']]}: {detection['confidence']:.2f}"
        
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame