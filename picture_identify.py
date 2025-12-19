import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = os.path.join('assets', 'garbage_classifier_final.keras')
IMG_SIZE = 384
LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# ==========================================
# 2. Load Model
# ==========================================
print("Loading model... please wait.")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

def predict_image():
    # Hide the main tkinter window
    Tk().withdraw()
    
    # Open file dialog
    print("Please select an image file...")

    # UPDATED: Added "All Files" (*.*) so you can see everything
    filename = askopenfilename(
        title="Choose your image",
        filetypes=[
            ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
            ("All Files", "*.*") 
        ]
    )
    
    if not filename:
        print("No file selected.")
        return

    # ==========================================
    # 3. Preprocessing
    # ==========================================
    # Load image using OpenCV
    img_bgr = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to what the model expects
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Add batch dimension (1, 384, 384, 3)
    input_data = np.expand_dims(img_resized, axis=0)
    
    # EfficientNet preprocessing
    input_data = preprocess_input(input_data)

    # ==========================================
    # 4. Prediction
    # ==========================================
    preds = model.predict(input_data)
    score = preds[0]
    
    # Get the top prediction
    top_class_index = np.argmax(score)
    top_class_label = LABELS[top_class_index]
    top_class_prob = score[top_class_index]

    # ==========================================
    # 5. Visualization
    # ==========================================
    plt.figure(figsize=(10, 5))

    # Plot 1: The Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {top_class_label}\nConfidence: {top_class_prob*100:.2f}%")
    plt.axis('off')

    # Plot 2: The Bar Chart (All probabilities)
    plt.subplot(1, 2, 2)
    bars = plt.barh(LABELS, score * 100, color='skyblue')
    
    # Highlight the winner
    bars[top_class_index].set_color('green')
    
    plt.xlabel('Confidence (%)')
    plt.title('Class Probabilities')
    plt.xlim(0, 100)
    
    # Add percentage text to bars
    for i, v in enumerate(score):
        plt.text(v*100 + 1, i, f"{v*100:.1f}%", va='center')

    plt.tight_layout()
    plt.show()

# Run the function
if __name__ == "__main__":
    while True:
        predict_image()
        # Ask to run again
        cont = input("Do you want to classify another image? (y/n): ")
        if cont.lower() != 'y':
            break
