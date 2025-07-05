# COMSYS-2025 - The 6th International Conference on Frontiers in Computing and Systems (COMSYS-2025)
# Theme: Robust Face Recognition and Gender Classification under Adverse Visual Conditions

## Task A: RCG-Net - ResNet50-Based Gender Classification Network
RCG-Net presents a sophisticated deep learning architecture for binary gender classification from facial images, leveraging transfer learning principles with advanced attention mechanisms. The model employs ResNet50 as its backbone network, pre-trained on ImageNet with frozen weights to serve as a robust feature extractor for 224×224×3 RGB input images. The architecture is enhanced with two complementary attention mechanisms: a Squeeze-and-Excitation (SE) block that performs channel-wise attention with a reduction ratio of 16, and a Convolutional Block Attention Module (CBAM) that combines both channel and spatial attention using 7×7 convolutions with global average and max pooling operations.
The network architecture follows a sequential flow: input images are processed through the frozen ResNet50 backbone, followed by a 64-filter convolutional layer with ReLU activation, then the SE block for channel attention, CBAM for dual attention, another 32-filter convolutional layer, global average pooling, a 64-unit dense layer with dropout regularization (0.4), and finally a single-unit sigmoid output layer for binary classification. The model is trained using the Adam optimizer with binary cross-entropy loss over 100 epochs with a batch size of 16, where labels are encoded as 0 for female and 1 for male classifications.
Data preprocessing follows ResNet50 specifications with mean centering and scaling, while comprehensive evaluation includes confusion matrix analysis, classification reports with precision/recall/F1-score metrics, and training curve visualization. The dual attention mechanism significantly enhances feature selection by emphasizing important channels through SE blocks while simultaneously focusing on relevant spatial regions via CBAM, resulting in improved classification accuracy. This architecture demonstrates the effectiveness of combining established CNN backbones with modern attention mechanisms for robust gender classification tasks in computer vision applications.

### To run the setup, simply run the RCG-NET Jupyter notebook to obtain the Resgendernet.h5 file (which is too large to upload to GitHub). This file enables the test script (testscript.py) to run the test dataset. Or the models are uploaded on google drive, in a text file "Download The Models" file. 

### Architecture Flow Diagram
```
Input (224×224×3) → ResNet50 (Frozen) → Conv2D(64) → SE Block → CBAM → Conv2D(32) → GAP → Dense(64) → Dropout(0.4) → Dense(1, Sigmoid) → Gender Prediction
```
## The Scores on the validation set:
 ```
              precision    recall  f1-score   support

      Female       0.84      0.72      0.78       105
        Male       0.91      0.95      0.93       317

    accuracy                           0.90       422
   macro avg       0.87      0.84      0.85       422
weighted avg       0.89      0.90      0.89       422
```

![image](https://github.com/user-attachments/assets/18e0c9a7-2d6a-4803-8654-0218e8ff7181)

![image](https://github.com/user-attachments/assets/b61fdf41-09bc-4b35-8122-185bd19dcbdd)

![image](https://github.com/user-attachments/assets/21127d7e-146b-470c-b91e-5fea95654063)

## Task B: TRF-Net: Triplet ResNet Fusion Network for Robust Face Recognition

**TRF-Net** is a face recognition framework built for real-world scenarios where image quality can be degraded due to environmental or acquisition issues. It combines the strength of transfer learning with advanced attention mechanisms and distance-based metric learning for high-precision identity verification and classification.

The architecture is based on a **frozen ResNet50** backbone pre-trained on ImageNet for feature extraction, followed by **custom convolutional layers** enhanced with **Squeeze-and-Excitation (SE)** blocks and **Convolutional Block Attention Modules (CBAM)**. These attention blocks allow the network to focus on the most discriminative facial features by applying both **channel-wise** and **spatial** attention refinement.

TRF-Net is trained using a **triplet loss function** with a margin of `0.2`, where each training sample consists of:

* an **anchor** image (usually a clean image),
* a **positive** image (a distorted version of the same person), and
* a **negative** image (an image of a different person).

This training strategy teaches the network to **minimize intra-class distance** (anchor-positive) and **maximize inter-class distance** (anchor-negative) within a learned **128-dimensional L2-normalized embedding space**.

All images are resized to **224×224 RGB** and normalized for input. During inference, TRF-Net computes embeddings for both **reference identities** (enrolled faces) and **query/test images**. Identification is performed via **cosine similarity**, where a test image is matched to the reference identity with the highest similarity score.

A robust test pipeline has been designed where:

* One image per identity is used to create reference embeddings.
* Test images (including unseen identities) are matched against the reference database.
* Top-1 prediction and **macro-averaged F1 scores** are computed to evaluate performance.

Optionally, a similarity **threshold** can be applied to reject unknown faces (e.g., when similarity is too low).

TRF-Net demonstrates **strong generalization** in handling image distortions such as **blur, rain, low resolution**, etc., and is well-suited for real-time deployment scenarios like surveillance, access control, and identity verification in unconstrained environments.

### Instructions:
To implement TRF-Net locally, first clone the repository and ensure you have Python 3.8+ installed with TensorFlow, OpenCV, and scikit-learn dependencies (you can install them using pip install -r requirements.txt). The trained face embedding model (.h5 file) with custom SE and CBAM layers should be placed in the project root or specified in testscript.py. Organize your data into reference and test folders with each identity in its own subfolder containing face images. Then run the evaluation using python testscript.py --model best_triplet_model.h5(Download this from the "download the model" file) --reference ./reference_data --test ./test_data to compute embeddings, perform cosine similarity-based matching, and display classification results. Make sure the input images are RGB face crops resized to 224×224 for best results.

### Architecture Flow Diagram
![image](https://github.com/user-attachments/assets/ba2201e1-4f59-42be-b185-4508a0dae484)
## The Scores on the validation set:
![download](https://github.com/user-attachments/assets/0cbe1545-015c-403d-9c78-c9576166aaac)

```
Top-1 Accuracy: 60.43%
Macro F1 Score: 0.7434
```
