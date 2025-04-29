import cv2
import numpy as np
import joblib
from scipy.special import legendre
from datetime import datetime
import time

class LiveObjectDetector:
    def __init__(self, model_path):
        # Load model components
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.metadata = model_data['metadata']
            print(f"Model loaded with {len(self.metadata['class_mapping'])} classes")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

        # Detection parameters
        self.min_confidence = 0.6  # Increased confidence threshold
        self.stable_detection_threshold = 3
        self.current_detections = {}
        self.detection_history = []
        self.last_detected_class = None
        self.last_detection_time = 0
        self.cooldown_period = 1  # seconds
        
        # Initialize camera with multiple attempts
        self.cap = None
        for i in range(3):  # Try different camera indices
            self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
            if self.cap.isOpened():
                print(f"Camera found at index {i}")
                # Set camera resolution for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                break
            else:
                if self.cap:
                    self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Could not access any camera (tried indices 0-2)")

    def __del__(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def handle_image_channels(self, img):
        """Preprocess image for feature extraction"""
        try:
            target_size = (self.metadata['target_w'], self.metadata['target_h'])
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            if len(resized_img.shape) == 3:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            return resized_img.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Image processing error: {e}")
            return None

    def extract_features(self, img):
        """Extract FFT and Legendre features"""
        try:
            gray = self.handle_image_channels(img)
            if gray is None:
                return None
                
            # FFT Features
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            fft_features = np.sort(magnitude.flatten())[::-1][:self.metadata['fft_top']]
            
            # Legendre Moments
            size = self.metadata['target_w']
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            moments = []
            
            for n in range(self.metadata['legendre_order'] + 1):
                for m in range(self.metadata['legendre_order'] + 1 - n):
                    Pn = legendre(n)(x)
                    Pm = legendre(m)(y)
                    moments.append(np.sum(gray * np.outer(Pm, Pn)))
            
            return np.concatenate([fft_features, np.array(moments)])
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def process_frame(self, frame):
        """Process frame and return detection results"""
        try:
            features = self.extract_features(frame)
            if features is None:
                return None
                
            features_scaled = self.scaler.transform([features])
            proba = self.model.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(proba)
            
            return {
                'class': self.metadata['class_mapping'][pred_idx],
                'confidence': proba[pred_idx],
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'valid': proba[pred_idx] >= self.min_confidence
            }
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None

    def should_log_detection(self, class_name):
        """Determine if detection should be logged"""
        current_time = time.time()
        
        # Don't log if same class was recently detected
        if (class_name == self.last_detected_class or current_time - self.last_detection_time < self.cooldown_period):
            return False
        
        self.last_detected_class = class_name
        self.last_detection_time = current_time
        return True

    def find_object_contour(self, frame):
        """Find the main object contour for better bounding box"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_contour)
        return None

    def draw_detection(self, frame, result):
        """Draw elegant detection information and tight bounding box"""
        height, width = frame.shape[:2]
        
        # Find object contour for tight bounding box
        bbox = self.find_object_contour(frame)
        
        # Draw bounding box if found
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # Fallback to centered box if no contour found
            cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
        
        # Create clean info panel
        info_panel = np.zeros((100, width, 3), dtype=np.uint8)
        
        if result and result['valid']:
            text = f"Detected: {result['class']} ({result['confidence']:.1%})"
            cv2.putText(info_panel, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(info_panel, "Scanning...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Combine frame with info panel
        return np.vstack([frame, info_panel])

    def run(self):
        """Main detection loop with clean UI"""
        print("Starting detection - Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame capture error")
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Handle valid detections
                if result and result['valid']:
                    if self.should_log_detection(result['class']):
                        print(f"{result['timestamp']} - {result['class']} ({result['confidence']:.1%})")
                
                # Display results
                display_frame = self.draw_detection(frame, result)
                cv2.imshow("Object Detector", display_frame)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            print("\nDetection session ended")

if __name__ == "__main__":
    try:
        detector = LiveObjectDetector("models/svm_model.joblib")
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}")