import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = os.path.join('assets', 'garbage_classifier_final.keras')
IMG_SIZE = 384  # Must match the size used during training
CONFIDENCE_THRESHOLD = 0.60  # Only show result if model is 60% sure

# Define the labels in the EXACT order of your training generator
# (Alphabetical order is standard for Keras flow_from_directory)
LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Colors for bounding boxes (B, G, R)
COLORS = {
    'Cardboard': (0, 165, 255), # Orange
    'Glass': (255, 255, 0),     # Cyan
    'Metal': (192, 192, 192),   # Gray
    'Paper': (255, 255, 255),   # White
    'Plastic': (0, 0, 255),     # Red
    'Trash': (0, 255, 0)        # Green
}

# ==========================================
# 2. Load Model
# ==========================================
print("Loading model... (this might take a moment)")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==========================================
# 3. Webcam Loop
# ==========================================
# 0 usually refers to the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Starting camera... Press 'q' to quit.")

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 2. Preprocess frame for the model
    # Convert BGR (OpenCV) to RGB (TensorFlow)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to the model's expected input size
    resized = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
    
    # Expand dims to create a batch of 1: (1, 384, 384, 3)
    input_data = np.expand_dims(resized, axis=0)
    
    # Apply EfficientNet preprocessing
    input_data = preprocess_input(input_data)

    # 3. Prediction
    preds = model.predict(input_data, verbose=0)
    score = np.max(preds)
    idx = np.argmax(preds)
    label = LABELS[idx]

    # 4. Display Logic
    # We define a "Region of Interest" (ROI) in the center of the screen
    # to guide the user where to hold the trash
    height, width, _ = frame.shape
    box_size = 300
    x1 = int(width / 2 - box_size / 2)
    y1 = int(height / 2 - box_size / 2)
    x2 = int(width / 2 + box_size / 2)
    y2 = int(height / 2 + box_size / 2)

    # Draw the targeting box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Determine display text based on confidence
    if score > CONFIDENCE_THRESHOLD:
        text = f"{label} ({score*100:.1f}%)"
        color = COLORS.get(label, (0, 255, 0))
        
        # Draw background bar for text
        cv2.rectangle(frame, (x1, y1-40), (x2, y1), color, -1)
        # Draw text
        cv2.putText(frame, text, (x1 + 10, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        # If unsure
        cv2.putText(frame, "Waiting for object...", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Show the frame
    cv2.imshow('Garbage Classifier (EfficientNetB0)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
