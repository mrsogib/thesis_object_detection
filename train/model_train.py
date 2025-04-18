import cv2
import numpy as np
import scipy.fftpack
import joblib
import os
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to extract features using Fourier Transform
def extract_fourier_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.resize(gray, (32, 32))  # Resize image to 32x32 to reduce memory usage
    f_transform = scipy.fftpack.fft2(small_image)
    f_shift = scipy.fftpack.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1).astype(np.float32)  # Use float32
    return magnitude_spectrum.flatten()

# Function to generate Legendre Polynomial Features
def legendre_polynomial_features(data, degree=4):
    coeffs = np.polynomial.legendre.legval(data, np.ones(degree))
    return coeffs.flatten()

# Collect custom dataset
def load_custom_dataset(dataset_path):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            label_map[class_name] = label_counter
            label_counter += 1
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                print(f"Loading image: {img_path}")

                img = cv2.imread(img_path)
                if img is not None:
                    fourier_features = extract_fourier_features(img)
                    legendre_features = legendre_polynomial_features(fourier_features)
                    features = np.hstack((fourier_features, legendre_features))  # Combine features
                    images.append(features)
                    labels.append(class_name)  # Use class name as label
                else:
                    print(f"Failed to load image: {img_path}")

    if len(images) == 0:
        print("No images were loaded. Please check the dataset path and image files.")
    else:
        print(f"Loaded {len(images)} images with {len(labels)} labels.")

    return np.array(images), np.array(labels), label_map

# Train SVM Classifier
def train_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))  # Ensure float32
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(X_train, y_train)
    
    accuracy = svm_classifier.score(X_test, y_test)
    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")
    
    return svm_classifier, scaler, X_scaled, y

# Main Execution
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/images')
X, y, label_map = load_custom_dataset(dataset_path)

# Check if data is loaded correctly
print(f"Loaded {len(X)} images with {len(y)} labels")

if len(X) > 0 and len(y) > 0:
    svm_model, scaler, X_scaled, y_scaled = train_svm(X, y)

    # Create the models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save the trained model and scaler using joblib
    joblib.dump(svm_model, os.path.join(models_dir, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

else:
    print("No valid data to train the model.")
