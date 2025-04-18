import cv2
import numpy as np
import joblib
import scipy.fftpack
import time
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the trained model and scaler
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Function to extract features using Fourier Transform
def extract_fourier_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.resize(gray, (32, 32))  # Use the same size as used during training
    f_transform = scipy.fftpack.fft2(small_image)
    f_shift = scipy.fftpack.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1).astype(np.float32)  # Use float32
    return magnitude_spectrum.flatten()

# Function to generate Legendre Polynomial Features
def legendre_polynomial_features(data, degree=4):
    coeffs = np.polynomial.legendre.legval(data, np.ones(degree))
    return coeffs.flatten()

# Real-time Object Detection via PC Camera
def real_time_object_detection():
    cap = cv2.VideoCapture(0)  # Open the default camera (0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features from the captured frame
        fourier_features = extract_fourier_features(frame)
        legendre_features = legendre_polynomial_features(fourier_features)
        features = np.hstack((fourier_features, legendre_features))  # Combine features

        # Normalize features
        features_scaled = scaler.transform([features])

        # Predict with SVM model
        prediction = svm_model.predict(features_scaled)
        prediction_proba = svm_model.predict_proba(features_scaled)
        confidence = prediction_proba[0][svm_model.classes_.tolist().index(prediction[0])] * 100

        # Print prediction, confidence, and timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Prediction: {prediction[0]}")

        # Draw rectangle around the object (entire frame in this case)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 2)

        # Add prediction text
        cv2.putText(frame, f'Prediction: {prediction[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Object Detection', frame)

        # Wait for 1 second before processing the next frame
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time object detection
real_time_object_detection()
