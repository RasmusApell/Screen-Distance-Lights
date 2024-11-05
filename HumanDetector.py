import cv2
import mediapipe as mp
import numpy as np
import time
import serial

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("Serial connection established")
            time.sleep(2)
        except serial.SerialException as e:
            print(f"Failed to open serial port. Error: {e}")
            self.ser = None
        
        self.KNOWN_FACE_WIDTH = 0.16 # My face width (m)
        self.FOCAL_LENGTH = 930      # Calibrated based on laptop camera Acer Swift

        self.last_serial_time = time.time()
        self.SERIAL_INTERVAL = 0.1 # Max serial output 10 Hz
        
    def send_distance(self, distance):
        if self.ser and (time.time() - self.last_serial_time) >= self.SERIAL_INTERVAL:
            try:
                dist_str = f"{distance:.2f}\n" if distance != float('inf') else "-1\n"
                self.ser.write(dist_str.encode())
                self.last_serial_time = time.time()
            except serial.SerialException as e:
                print(f"Serial communication error: {e}")
    
    def estimate_distance(self, face_width_relative):
        # Convert relative width to pixels (relative * image width)
        frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        pixel_width = face_width_relative * frame_width
        
        distance = (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / pixel_width
        return distance
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_detection.process(rgb_frame)
        
        min_distance = float('inf')
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to pixel coordinates
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                distance = self.estimate_distance(bbox.width)
                
                if distance < min_distance:
                    min_distance = distance
                
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                distance_text = f"Distance: {distance:.2f}m"
                cv2.putText(frame, distance_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                confidence_text = f"Confidence: {detection.score[0]:.2f}"
                cv2.putText(frame, confidence_text, (x, y + height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, min_distance
    
    def run(self):
        print("\nControls:")
        print("'q' - Quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame, min_distance = self.process_frame(frame)
            self.send_distance(min_distance)            

            cv2.imshow('Face Detection', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        if self.ser:
            self.ser.close()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()
