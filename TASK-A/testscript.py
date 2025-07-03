# ===========================
# Imports
# ===========================
import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from custom_layers import CBAM, se_block  # Custom attention modules

# Register custom layers for model loading
custom_objects = {
    'CBAM': CBAM,
    'se_block': se_block
}

# ===========================
# ⚙️ Configuration
# ===========================
MODEL_PATH = 'ResGenderNet.h5'    # Path to trained model
TEST_DIR = 'val'                  # Folder containing validation images (Male/, Female/)
IMG_SIZE = (224, 224)             # ResNet input size
BATCH_SIZE = 32                   # Batch size for inference

# ===========================
#  Load Trained Model
# ===========================
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
print("Model loaded successfully.")

# ===========================
#  Image Preprocessing
# ===========================
def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.resnet50.preprocess_input(image)  # Normalize for ResNet50
    return image

# ===========================
#  Load Test Dataset
# ===========================
male_paths = glob.glob(os.path.join(TEST_DIR, 'Male', '*.jpg'))
female_paths = glob.glob(os.path.join(TEST_DIR, 'Female', '*.jpg'))

all_paths = female_paths + male_paths
all_labels = [0] * len(female_paths) + [1] * len(male_paths)  # 0: Female, 1: Male

# Preprocess and stack all test images
all_images = [preprocess_image(p) for p in all_paths]
all_images = tf.stack(all_images)
all_labels = tf.convert_to_tensor(all_labels, dtype=tf.int32)

# ===========================
# Run Inference
# ===========================
y_pred_probs = model.predict(all_images, batch_size=BATCH_SIZE)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()  # Binary threshold
y_true = all_labels.numpy()

# ===========================
# Metrics and Evaluation
# ===========================
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Female", "Male"]))

# ===========================
#  Visualize Predictions
# ===========================
def show_sample_predictions(n=5):
    for i in range(n):
        image = all_images[i].numpy()
        true_label = "Male" if y_true[i] == 1 else "Female"
        pred_label = "Male" if y_pred[i] == 1 else "Female"
        
        # Convert ResNet preprocessed image back to displayable format
        plt.imshow((image + 1) * 127.5 / 255.0)
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.axis('off')
        plt.show()

show_sample_predictions(5)

# ===========================
# Confusion Matrix
# ===========================
cm = confusion_matrix(y_true, y_pred)
print("\n📊 Confusion Matrix:\n", cm)

# ===========================
# Model Weights Inspection
# ===========================
for layer in model.layers:
    print(f"\n🔹 Layer: {layer.name}")
    weights = layer.get_weights()
    if weights:
        for i, w in enumerate(weights):
            print(f"  Weight {i + 1} shape: {w.shape}")
            print(w)  # ⚠️ May print large arrays
    else:
        print("  No trainable weights.")

# ===========================
# Summary of Trainable Parameters
# ===========================
print("\n Trainable Weights Summary:")
for weight in model.trainable_weights:
    print(f"{weight.name} — shape: {weight.shape}")

trainable_params = np.sum([np.prod(w.shape) for w in model.trainable_weights])
print(f"\n Total trainable parameters: {trainable_params}")
