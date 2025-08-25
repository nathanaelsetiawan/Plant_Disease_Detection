# Convolutional Neural Network for Tomato Leaf Disease Detection
This mini project applies the Convolutional Neural Network (CNN) algorithm to classify whether a tomato leaf is healthy or affected by a disease.

## Objective
The main goal of this project is to implement a simple deep learning classifier (CNN) and evaluate its performance on a tomato leaf dataset.

## Dataset
The dataset used in this project is a subset of the **[Plant Village](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)** dataset available on Kaggle.

From the full dataset, only 3 folders were selected for this project, specifically related to tomato leaves:
- `Tomato_Late_blight`
- `Tomato_Early_blight`
- `Tomato_healthy`

This subset was chosen to focus on classifying tomato leaves as either healthy or diseased.

### Dataset Statistics
| Class                    | Number of Images |
|--------------------------|------------------|
| Tomato_Late_blight       | 1,909            |
| Tomato_Early_blight      | 1,000            |
| Tomato_healthy           | 1,591            |
| **Total**                | **4,500**        |

## Model Development
The model was developed using Convolutional Neural Network (CNN).

### Steps in Model Development
**1. Data Preprocessing**
   - Resized all images into a uniform dimension (224 x 224).
   - Set the batch size into 32.
   - Split the dataset into train sets (80%), test sets (10%), and validation sets (10%).

**2. Model Architecture**
   - Base Model: VGG16 pre-trained on ImageNet with `include_top=False`.  
     - The convolutional layers were **frozen** to retain learned features.  
   - Custom Layers were added on top:  
     - `GlobalAveragePooling2D` to reduce dimensions while keeping spatial information.  
     - `Dense (128 units, ReLU)` for feature learning.  
     - `Dropout (0.5)` to prevent overfitting.  
     - `Dense (3 units, Softmax)` as the output layer for classifying tomato leaves into 3 categories.
  
**3. Training**
   - The model was compiled with:  
     - Optimizer: Adam  
     - Loss Function: Sparse Categorical Crossentropy  
     - Metric: Accuracy  
   - Trained for 20 epochs with a batch size of 32, using the training set for learning and the validation set for performance monitoring.
   - Training history showed smooth convergence:
     - Training and validation accuracy stabilized at around 98–99%.
     - Training and validation loss decreased steadily with no indication of overfitting.
  <p align="center">
    <img width="680" height="682" alt="Unknown" src="https://github.com/user-attachments/assets/9190eebe-7fb0-4088-8f95-c5796027e196" />
  </p>

**4. Model Evaluation**
   - On the test set, the model achieved an overall accuracy of 99%.
   - Classification Report:
     - Tomato_Early_blight: Precision = 1.00, Recall = 0.96, F1 = 0.98 
     - Tomato_Late_blight: Precision = 0.98, Recall = 1.00, F1 = 0.99
     - Tomato_healthy: Precision = 0.99, Recall = 1.00, F1 = 1.00
   - Results indicate the model is highly reliable with only minor misclassifications, mostly in distinguishing Early blight.
  <p align="center">  
    <img width="497" height="157" alt="Screenshot 2025-08-25 at 17 40 33" src="https://github.com/user-attachments/assets/66f995b5-769b-4ef9-bfbd-c037009c90d6" />
  </p>

## How to Run the Notebook
1. Open the notebook in Google Colab.
2. Run all cells in order (Runtime > Run all).
3. The final section will display evaluation metrics and prediction results on the test set.
