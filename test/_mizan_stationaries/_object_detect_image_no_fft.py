import cv2
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import glob
from datetime import datetime

# Load the trained model and scaler
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Function to extract basic features (ensuring the same number of features as during training)
def extract_basic_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.resize(gray, (64, 32))  # Adjust the size to match the training setup
    features = small_image.flatten()
    return features

# Path to the directory containing the images
image_dir = "../test_images"

# Retrieve all image file paths from the directory
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

# Initialize a dictionary to store the detections for each image
image_detections = {}

# Process each image in the directory
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is not None:
        # Extract basic features from the image
        features = extract_basic_features(image)

        # Normalize features
        features_scaled = scaler.transform([features])

        # Predict with SVM model
        prediction = svm_model.predict(features_scaled)
        prediction_proba = svm_model.predict_proba(features_scaled)
        confidence = prediction_proba[0][svm_model.classes_.tolist().index(prediction[0])] * 100

        # Print prediction in real time
        print(f"Image: {image_name} | Prediction: {prediction[0]} | Confidence: {confidence:.2f}%")

        # Store the prediction for the image
        if image_name in image_detections:
            image_detections[image_name] += f", {prediction[0]} ({confidence:.2f}%)"
        else:
            image_detections[image_name] = f"{prediction[0]} ({confidence:.2f}%)"
    else:
        print(f"Failed to load image: {image_path}")

current_time = datetime.now().strftime('%Y%m%d_%H%M%S%f')
# Write the predictions to a text file
with open(f'mine_no_fft_{current_time}.txt', "w") as f:
    for image_name, prediction in image_detections.items():
        f.write(f"{image_name} : {prediction}\n")

print(f'\nBulk image detection completed and results saved to mine_no_fft_{current_time}.txt\n')
