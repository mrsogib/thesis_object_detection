import os
import joblib
import numpy as np
from skimage import io, transform, color
from scipy.special import legendre
from datetime import datetime
from tqdm import tqdm

def load_model(model_path):
    """Load trained model and metadata"""
    data = joblib.load(model_path)
    return data['model'], data['scaler'], data['metadata']

def handle_image_channels(img, metadata):
    """Resize and convert to grayscale using model parameters"""
    target_size = (metadata['target_h'], metadata['target_w'])
    resized_img = transform.resize(img, target_size, anti_aliasing=True)
    
    if resized_img.ndim == 2:
        return resized_img
    elif resized_img.shape[2] >= 3:
        return color.rgb2gray(resized_img[:, :, :3])
    return color.rgb2gray(resized_img)

def extract_fft_features(gray_img, metadata):
    """FFT feature extraction using model parameters"""
    fft = np.fft.fft2(gray_img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    sorted_mag = np.sort(magnitude.flatten())[::-1]
    return sorted_mag[:metadata['fft_top']]

def extract_legendre_moments(gray_img, metadata):
    """Legendre moments using model parameters"""
    size = metadata['target_w']
    order = metadata['legendre_order']

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    moments = []
    
    for n in range(order + 1):
        for m in range(order + 1 - n):
            Pn = legendre(n)(x)
            Pm = legendre(m)(y)
            moments.append(np.sum(gray_img * np.outer(Pm, Pn)))
    return np.array(moments)

def process_image(img_path, scaler, metadata):
    """Process single image for prediction"""
    try:
        img = io.imread(img_path)
        gray = handle_image_channels(img, metadata)
        fft_features = extract_fft_features(gray, metadata)
        legendre_features = extract_legendre_moments(gray, metadata)
        features = np.concatenate([fft_features, legendre_features])
        features_scaled = scaler.transform([features])
        return features_scaled, None
    except Exception as e:
        return None, str(e)


def bulk_predict(image_dir, model_path):
    """Predict classes for all images in directory"""
    model, scaler, metadata = load_model(model_path)
    class_mapping = metadata['class_mapping']
    
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    high_confidence_count = 0

    for filename in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(image_dir, filename)
        features, error = process_image(img_path, scaler, metadata)
        
        if features is not None:
            proba = model.predict_proba(features)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
            
            if confidence >= 0.5:
                high_confidence_count += 1
                
            results.append({
                'file': filename,
                'class': class_mapping[pred_idx],
                'confidence': confidence,
                'error': None
            })
        else:
            results.append({
                'file': filename,
                'class': None,
                'confidence': None,
                'error': error
            })

    # Save results with UTF-8 encoding
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"prediction_results_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"High Confidence Detections (>=50%): {high_confidence_count}/{len(results)} ({high_confidence_count/len(results):.1%})\n")
        f.write("Image Path|Predicted Class|Confidence|Error\n")
        f.write("-"*50 + "\n")
        
        for res in results:
            conf_str = f"{res['confidence']:.2%}" if res['confidence'] is not None else "N/A"
            line = f"{res['file']}|{res['class']}|{conf_str}|{res['error'] or ''}\n"
            f.write(line)

    print(f"\nPrediction complete! Results saved to {output_file}")
    print(f"High Confidence Detections: {high_confidence_count} ({high_confidence_count/len(results):.1%})")
    return results

if __name__ == "__main__":
    MODEL_PATH = "models/svm_model.joblib"
    IMAGE_DIR = "../test_images"
    results = bulk_predict(IMAGE_DIR, MODEL_PATH)

