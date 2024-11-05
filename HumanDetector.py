import cv2
import mediapipe as mp
import numpy as np
import time
import serial

class FaceDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for closer faces, 1 for faces further away
            min_detection_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Initialize serial connection
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("Serial connection established")
            time.sleep(2)
        except serial.SerialException as e:
            print(f"Failed to open serial port. Error: {e}")
            self.ser = None
        
        # Known parameters for distance estimation
        self.KNOWN_FACE_WIDTH = 0.16  # Average face width in meters
        self.FOCAL_LENGTH = 930      # Will be calibrated
        
        # For smoothing distance estimates
        self.distance_history = []
        self.HISTORY_SIZE = 5
        
        # For serial communication
        self.last_serial_time = time.time()
        self.SERIAL_INTERVAL = 0.1
        
    def send_distance(self, distance):
        """Send distance data over serial"""
        if self.ser and (time.time() - self.last_serial_time) >= self.SERIAL_INTERVAL:
            try:
                dist_str = f"{distance:.2f}\n" if distance != float('inf') else "-1\n"
                self.ser.write(dist_str.encode())
                self.last_serial_time = time.time()
            except serial.SerialException as e:
                print(f"Serial communication error: {e}")
    
    def estimate_distance(self, face_width_relative):
        """Estimate distance using relative face width"""
        # Convert relative width to pixels (relative * image width)
        frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        pixel_width = face_width_relative * frame_width
        
        # Calculate distance
        distance = (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / pixel_width
        
        # Smooth the distance estimate
        self.distance_history.append(distance)
        if len(self.distance_history) > self.HISTORY_SIZE:
            self.distance_history.pop(0)
        
        return sum(self.distance_history) / len(self.distance_history)
    
    def process_frame(self, frame):
        """Process a single frame and return the annotated frame with distance estimation"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_detection.process(rgb_frame)
        
        min_distance = float('inf')
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to pixel coordinates
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Calculate distance
                distance = self.estimate_distance(bbox.width)
                
                # Update minimum distance
                if distance < min_distance:
                    min_distance = distance
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Add distance text
                distance_text = f"Distance: {distance:.2f}m"
                cv2.putText(frame, distance_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add confidence score
                confidence_text = f"Confidence: {detection.score[0]:.2f}"
                cv2.putText(frame, confidence_text, (x, y + height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Send distance over serial
        self.send_distance(min_distance)
        
        return frame, min_distance
    
    def run(self):
        """Main loop for capturing and processing video"""
        print("\nControls:")
        print("'q' - Quit")
        print("'c' - Calibrate")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, min_distance = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Face Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\nCalibration Mode:")
                print("Stand exactly 1 meter from camera")
                print("Press 's' when ready")
                
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_detection.process(rgb_frame)
                    
                    if results.detections:
                        bbox = results.detections[0].location_data.relative_bounding_box
                        frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        pixel_width = bbox.width * frame_width
                        self.FOCAL_LENGTH = (pixel_width * 1.0) / self.KNOWN_FACE_WIDTH
                        
                        # Show current measurements
                        cv2.putText(frame, f"Width: {pixel_width:.0f}px", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Focal Length: {self.FOCAL_LENGTH:.0f}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow('Calibration', frame)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        print(f"Calibration saved! Focal length: {self.FOCAL_LENGTH:.0f}")
                        break
        
        # Cleanup
        if self.ser:
            self.ser.close()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()