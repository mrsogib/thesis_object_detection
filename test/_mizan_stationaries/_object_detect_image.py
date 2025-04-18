import cv2
import numpy as np
import joblib
import scipy.fftpack
import os
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

# Directory containing images for detection
image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_images')
output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output_images')
os.makedirs(output_directory, exist_ok=True)

# Initialize a dictionary to store the detections for each image
image_detections = {}

# Process each image in the directory
for image_name in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_name)
    image = cv2.imread(image_path)
    if image is not None:
        # Extract features from the image
        fourier_features = extract_fourier_features(image)
        legendre_features = legendre_polynomial_features(fourier_features)
        features = np.hstack((fourier_features, legendre_features))  # Combine features

        # Normalize features
        features_scaled = scaler.transform([features])

        # Predict with SVM model
        prediction = svm_model.predict(features_scaled)
        prediction_proba = svm_model.predict_proba(features_scaled)
        confidence = prediction_proba[0][svm_model.classes_.tolist().index(prediction[0])] * 100

        # Print prediction in real time
        print(f"Image: {image_name} | Prediction: {prediction[0]} | Confidence: {confidence:.2f}%")

        # Draw rectangle around the object
        h, w, _ = image.shape
        cv2.rectangle(image, (0, 0), (w, h), (0, 255, 0), 2)
        
        # Add prediction text
        cv2.putText(image, f'Prediction: {prediction[0]} ({confidence:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        result_image_path = os.path.join(output_directory, image_name)
        cv2.imwrite(result_image_path, image)
        
        # Store the prediction for the image, separating multiple objects by commas
        if image_name in image_detections:
            image_detections[image_name] += f", {prediction[0]} ({confidence:.2f}%)"
        else:
            image_detections[image_name] = f"{prediction[0]} ({confidence:.2f}%)"
    else:
        print(f"Failed to load image: {image_path}")

current_time = datetime.now().strftime('%Y%m%d_%H%M%S%f')

# Write the predictions to a text file
with open(f'mine_{current_time}.txt', "w") as f:
    for image_name, prediction in image_detections.items():
        f.write(f"{image_name} : {prediction}\n")

print(f'\nObject detection completed and results saved to mine_{current_time}.txt\n')
