import cv2
import numpy as np
import time

class FaceDetector:
    def __init__(self):
        # Load the face detection classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Known parameters for distance estimation
        self.KNOWN_FACE_WIDTH = 0.15  # Average face width in meters
        self.FOCAL_LENGTH = 1000  # Will need calibration
        
        # For smoothing distance estimates
        self.distance_history = []
        self.HISTORY_SIZE = 5
        
    def estimate_distance(self, pixel_width):
        """Estimate distance using the formula: Distance = (Known Width x Focal Length) / Pixel Width"""
        distance = (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / pixel_width
        
        # Add to history and get smoothed value
        self.distance_history.append(distance)
        if len(self.distance_history) > self.HISTORY_SIZE:
            self.distance_history.pop(0)
        
        # Return smoothed distance
        return sum(self.distance_history) / len(self.distance_history)
        
    def calibrate_focal_length(self, known_distance, known_pixel_width):
        """Calibrate focal length using a known distance and measured pixel width"""
        self.FOCAL_LENGTH = (known_pixel_width * known_distance) / self.KNOWN_FACE_WIDTH
        print(f"Calibrated focal length: {self.FOCAL_LENGTH}")
        
    def process_frame(self, frame):
        """Process a single frame and return the annotated frame with distance estimation"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=8,
            minSize=(30, 30)
        )
        
        min_distance = float('inf')
        
        # Process detected faces
        for (x, y, w, h) in faces:
            # Estimate distance
            distance = self.estimate_distance(w)
            
            # Update minimum distance
            if distance < min_distance:
                min_distance = distance
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add distance text
            distance_text = f"Distance: {distance:.2f}m"
            cv2.putText(frame, distance_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add face width in pixels (useful for calibration)
            width_text = f"Width: {w}px"
            cv2.putText(frame, width_text, (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS counter
        fps_text = f"FPS: {int(1/(time.time() - self.last_time))}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.last_time = time.time()
        
        return frame, min_distance
    
    def run(self):
        """Main loop for capturing and processing video"""
        self.last_time = time.time()
        
        # Calibration mode flag
        calibration_mode = False
        known_distance = 1.0  # 1 meter for calibration
        
        print("Press 'c' to enter calibration mode")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, min_distance = self.process_frame(frame)
            
            # Display calibration instructions if in calibration mode
            if calibration_mode:
                cv2.putText(processed_frame, "CALIBRATION MODE: Stand 1m from camera", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Face Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                calibration_mode = True
                print("Calibration mode activated. Stand exactly 1 meter from the camera")
                print("Press 's' to save calibration")
            elif key == ord('s') and calibration_mode:
                # Use the current face width for calibration
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    # Correctly unpack the first face detection
                    x, y, w, h = faces[0]
                    self.calibrate_focal_length(known_distance, w)
                    calibration_mode = False
                    print("Calibration complete!")
                else:
                    print("No face detected for calibration!")
                
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()