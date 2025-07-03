# ============================
#  Imports
# ============================
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from custom_layers import CBAM, se_block  # Your custom attention modules

# ============================
#  Configuration
# ============================
MODEL_PATH = "face_embedding_model"  # Folder containing the trained model
IMG_SIZE = (224, 224)                # Image resize shape for model
THRESHOLD = 0.5                      # Cosine similarity threshold

# ============================
#  Load Model with Custom Layers
# ============================
custom_objects = {
    'CBAM': CBAM,
    'se_block': se_block
}
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
print(" Model with CBAM and SE loaded successfully.")

# ============================
# 🖼️ Image Preprocessing
# ============================
def preprocess_image(img_path):
    """
    Loads and preprocesses an image for inference.
    - Reads image using OpenCV
    - Resizes to IMG_SIZE
    - Normalizes to [0, 1]
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return img

# ============================
#  Embedding Extraction
# ============================
def get_embedding(img_path):
    """
    Returns the embedding vector of an image.
    """
    img = preprocess_image(img_path)
    embedding = model.predict(np.expand_dims(img, axis=0), verbose=0)
    return embedding

# ============================
#  Cosine Similarity Checker
# ============================
def is_same_person(emb1, emb2):
    """
    Computes cosine similarity and decides if two embeddings belong to the same person.
    """
    sim = cosine_similarity(emb1, emb2)[0][0]
    print(f"Cosine Similarity: {sim:.4f}")
    return sim > THRESHOLD

# ============================
# Main Execution Block
# ============================
if __name__ == "__main__":
    anchor_path = input("Enter path to anchor image: ").strip()
    test_path = input("Enter path to test image: ").strip()

    emb1 = get_embedding(anchor_path)
    emb2 = get_embedding(test_path)

    if is_same_person(emb1, emb2):
        print(" SAME PERSON")
    else:
        print(" DIFFERENT PERSON")
