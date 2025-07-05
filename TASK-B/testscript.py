# ============================
#  Imports
# ============================
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from custom_layers import CBAM, se_block
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

# ============================
#  Configuration
# ============================
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # You can tune this if using "unknown" logic

# ============================
#  Load Model with Custom Layers
# ============================
def load_embedding_model(model_path):
    custom_objects = {'CBAM': CBAM, 'se_block': se_block}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✅ Model with CBAM and SE loaded successfully.")
    return model

# ============================
#  Preprocess a Single Image
# ============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype("float32"))
    return img

# ============================
#  Generate Embedding
# ============================
def get_embedding(model, img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)[0]  # 128D

# ============================
#  Build Reference Embedding Dict
# ============================
def build_reference_embeddings(model, reference_folder):
    ref_dict = {}
    for person in os.listdir(reference_folder):
        person_folder = os.path.join(reference_folder, person)
        if not os.path.isdir(person_folder):
            continue
        image_files = os.listdir(person_folder)
        if not image_files:
            continue
        ref_img = os.path.join(person_folder, image_files[0])  # Take 1 reference image
        try:
            emb = get_embedding(model, ref_img)
            ref_dict[person] = emb
        except Exception as e:
            print(f"⚠️ Error processing {ref_img}: {e}")
    print(f"✅ Loaded {len(ref_dict)} reference embeddings.")
    return ref_dict

# ============================
#  Prediction Function
# ============================
def predict_identity(embedding, reference_dict):
    best_score = -1
    predicted_label = None
    for label, ref_emb in reference_dict.items():
        score = cosine_similarity([embedding], [ref_emb])[0][0]
        if score > best_score:
            best_score = score
            predicted_label = label
    return predicted_label, best_score

# ============================
#  Main Evaluation Function
# ============================
def evaluate(model, reference_dict, test_folder):
    y_true = []
    y_pred = []

    for person in os.listdir(test_folder):
        person_path = os.path.join(test_folder, person)
        if not os.path.isdir(person_path):
            continue
        for file in os.listdir(person_path):
            test_img = os.path.join(person_path, file)
            try:
                test_emb = get_embedding(model, test_img)
                pred_label, score = predict_identity(test_emb, reference_dict)
                y_true.append(person)
                y_pred.append(pred_label)
                print(f"🧪 {file} → Predicted: {pred_label} | True: {person} | Score: {score:.4f}")
            except Exception as e:
                print(f"❌ Error processing {test_img}: {e}")

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

# ============================
#  Script Entry Point
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Test Script with Embedding Comparison")
    parser.add_argument("--model", required=True, help="Path to trained .h5 model")
    parser.add_argument("--reference", required=True, help="Path to reference folder")
    parser.add_argument("--test", required=True, help="Path to test folder")
    args = parser.parse_args()

    model = load_embedding_model(args.model)
    reference_dict = build_reference_embeddings(model, args.reference)
    evaluate(model, reference_dict, args.test)
