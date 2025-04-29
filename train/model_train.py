import os
import joblib
import numpy as np
import cv2
from scipy.special import legendre
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from datetime import datetime
from collections import defaultdict

# -------------------------------
# 1. Feature Extraction (using OpenCV)
# -------------------------------
DIMENSION = 39
FFT_TOP = int(.12*DIMENSION*DIMENSION)
order = 11

def handle_image_channels(img):
    """Resize and convert to grayscale using OpenCV"""
    # Convert to RGB if it's BGR (OpenCV loads as BGR by default)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize using OpenCV
    resized_img = cv2.resize(img, (DIMENSION, DIMENSION), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale if needed
    if len(resized_img.shape) == 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    
    # Normalize to [0, 1] range
    resized_img = resized_img.astype(np.float32) / 255.0
    return resized_img

def extract_fft_features(gray_img):
    """Extract FFT features using OpenCV"""
    fft = np.fft.fft2(gray_img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    sorted_mag = np.sort(magnitude.flatten())[::-1]
    return sorted_mag[:FFT_TOP]

def extract_legendre_moments(gray_img):
    """Legendre moments using OpenCV"""
    x = np.linspace(-1, 1, DIMENSION)
    y = np.linspace(-1, 1, DIMENSION)
    moments = []
    for n in range(order + 1):
        for m in range(order + 1 - n):
            Pn = legendre(n)(x)
            Pm = legendre(m)(y)
            moments.append(np.sum(gray_img * np.outer(Pm, Pn)))
    return np.array(moments)

# -------------------------------
# 2. Train and Save Model
# -------------------------------
def train_and_save(root_dir):
    # Create models directory first to avoid any path issues
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    features, labels = [], []
    class_counts = defaultdict(int)
    image_records = []
    success_count = 0
    class_names = sorted(os.listdir(root_dir))
    
    print("\n=== Dataset Processing ===")
    
    # Process all images
    for class_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[cls_name] = len(images)
        
        for img_file in images:
            img_path = os.path.join(cls_dir, img_file)
            record = {
                'path': img_path,
                'actual_class': cls_name,
                'status': 'skipped',
                'error': None,
                'features': None
            }
            
            try:
                # Read image using OpenCV
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Could not read image {img_path}")
                
                gray = handle_image_channels(img)
                fft = extract_fft_features(gray)
                legendre = extract_legendre_moments(gray)
                
                features.append(np.concatenate([fft, legendre]))
                labels.append(class_idx)
                record.update({
                    'status': 'processed',
                    'features': features[-1]
                })
                success_count += 1
            except Exception as e:
                record['error'] = str(e)
            
            image_records.append(record)

    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total Classes: {len(class_names)}")
    print(f"Total Images: {len(image_records)}")
    print(f"Successful Images: {success_count}")    
    print(f"Skipped Images: {len(image_records) - success_count}")
    
    print("\n=== Class Distribution ===")
    for idx, cls_name in enumerate(class_names):
        print(f"Class {idx}: {cls_name} - {class_counts[cls_name]} images")

    # Check if we have any successful images
    if success_count == 0:
        raise ValueError("No images were successfully processed. Cannot train model.")

    # Split data
    success_indices = [i for i, r in enumerate(image_records) if r['status'] == 'processed']
    train_idx, val_idx = train_test_split(
        success_indices,
        test_size=0.2,
        stratify=[labels[i] for i in success_indices],
        random_state=42
    )

    # Create validation report
    report_path = os.path.join(models_dir, "validation_report.csv")
    
    # Train model
    X_train = [image_records[i]['features'] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [image_records[i]['features'] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)

    # Generate predictions for validation set
    val_probs = svm.predict_proba(X_val)
    val_preds = svm.predict(X_val)

    # Create validation report
    with open(report_path, 'w') as f:
        f.write("Image Path,Actual Class,Predicted Class,Confidence\n")
        for i, idx in enumerate(val_idx):
            record = image_records[idx]
            actual_class = record['actual_class']
            pred_class = class_names[val_preds[i]]
            confidence = val_probs[i][val_preds[i]]
            
            f.write(f"{record['path']},{actual_class},{pred_class},{confidence:.4f}\n")

    # Create metadata
    metadata = {
        "class_mapping": {idx: name for idx, name in enumerate(class_names)},
        "class_distribution": dict(class_counts),
        "target_w": DIMENSION,
        "target_h": DIMENSION,
        "fft_top": FFT_TOP,
        "legendre_order": order,
        "train_accuracy": svm.score(X_train, y_train),
        "val_accuracy": svm.score(X_val, y_val),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_report": report_path
    }

    # Save model
    model_path = os.path.join(models_dir, "svm_model.joblib")
    joblib.dump({"model": svm, "scaler": scaler, "metadata": metadata}, model_path)

    # Print summary
    print("\n=== Training Summary ===")
    print(f"Model Dimension: {DIMENSION}x{DIMENSION}")
    print(f"FFT Features: {FFT_TOP}")
    print(f"Legendre Order: {order}")
    print(f"Train Accuracy: {metadata['train_accuracy']:.2%}")
    print(f"Validation Accuracy: {metadata['val_accuracy']:.2%}")
    print(f"Validation report saved to: {report_path}")
    print(f"Model saved to: {model_path}")

# -------------------------------
# 3. Execution
# -------------------------------
if __name__ == "__main__":
    # Use absolute path to be safe
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")
    train_and_save(dataset_path)