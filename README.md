# Seizure Prediction with Deep Learning and Machine Learning

## Project Overview
This project focuses on predicting seizures using EEG data by leveraging both **deep learning** and **machine learning** models. The goal is to develop a system that can detect the onset of seizures, providing timely warnings for medical intervention. The dataset used consists of EEG recordings from pediatric subjects monitored for seizures.

### Key Models Explored:
- **Machine Learning Models**: Support Vector Machine (SVM), Random Forest (RF), and Gradient Boosted Model (GBM).
- **Deep Learning Model**: Convolutional Neural Network (CNN).

## Data Preprocessing
- **For Machine Learning**:
  - **Label Encoding**: Categorical labels were converted to numerical values.
  - **Normalization**: Features were normalized to ensure consistency across variables.
  - **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets.
  
- **For CNN**:
  - **Data Extraction**: Numerical values from EEG recordings were extracted.
  - **PyTorch Integration**: Data was converted into PyTorch tensors to facilitate seamless model integration.
  - **Normalization**: Mean and standard deviation were calculated for standardizing EEG values.

## 🧠 CNN Model Architecture
The core of the project lies in the **Convolutional Neural Network (CNN)**, which is particularly suited for detecting patterns in the EEG data:

- **Convolutional Layers**: 
  - 3 convolutional blocks with kernel sizes of 16, 8, and 4, respectively.
  - Stride settings of 4, 2, and 1 for extracting fine-grained patterns from the EEG data.
  - **Batch Normalization** and **Dropout** layers after each convolution block to enhance performance and prevent overfitting.

- **Fully Connected Layers**: 
  - Two fully connected layers that transition from convolutional operations to final decision-making, further refining the model's ability to classify seizure data.

- **Activation Functions**:
  - **ReLU** activation functions are used for non-linearity in the model.

- **Optimization**:
  - The model was trained using the **Adam optimizer** and **Cross Entropy loss function**, ensuring fast convergence and robust predictions.

## Training & Hyperparameter Tuning
- **Batch Processing**: EEG data was processed in batches to optimize model training efficiency.
- **GPU Acceleration**: Leveraged GPU resources to speed up training.
- **Hyperparameter Exploration**: Different channel configurations were experimented with to maximize accuracy, resulting in an optimal configuration of 4, 2, and 4 channels for the three convolution layers.

## Results
- The **CNN model** achieved an overall test accuracy of **74%**, outperforming the other machine learning models in classifying EEG recordings for seizure prediction.
- **Confusion Matrix**: The confusion matrix provided insights into model performance by showing true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Precision, recall, and F1-score metrics were computed, demonstrating the robustness of the CNN model in detecting seizures.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/seizure-prediction-with-dl-ml.git
