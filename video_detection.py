import cv2
from utils import display_detections, calculate_fps
import numpy as np

def process_video(video_path, net, classes, colors, conf_threshold=0.2):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                   0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        # Display results
        display_detections(frame, detections, classes, colors, conf_threshold)
        
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()