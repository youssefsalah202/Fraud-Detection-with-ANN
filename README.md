# Fraud-Detection-with-ANN
📌 Project Overview

This project applies Deep Learning to detect fraudulent transactions in the Fraud Detection Dataset. Originally, the implementation used a 1D Convolutional Neural Network (CNN), but the model was later adapted into a feed-forward Artificial Neural Network (ANN), which is better suited for tabular data.

The goal is to classify transactions as:

1 → Fraudulent

0 → Non-fraudulent

⚙️ Tech Stack

Python 3.10+

TensorFlow / Keras (Deep Learning framework)

scikit-learn (data preprocessing & metrics)

NumPy & Pandas (data manipulation)

Matplotlib (visualizations)

📂 Dataset

File: fraudTrain.csv

Contains 1,296,675 transactions and 11 numeric features.

Target column: is_fraud

Since fraud data is highly imbalanced, the dataset was balanced by downsampling non-fraud records to match the number of fraud records.

🔄 Data Preprocessing

Removed non-numeric columns.

Balanced dataset by downsampling majority class.

Split into training (80%) and testing (20%) using train_test_split.

Features were standardized using StandardScaler.

Converted labels to NumPy arrays.

🧠 ANN Model Architecture

The CNN was replaced with a fully connected ANN:

Input Layer: 11 features
↓
Dense(128, ReLU) + BatchNormalization + Dropout(0.3)
↓
Dense(64, ReLU) + BatchNormalization + Dropout(0.3)
↓
Dense(32, ReLU) + Dropout(0.2)
↓
Dense(1, Sigmoid)  → binary output (fraud / non-fraud)


Optimizer: Adam (lr = 0.0001)
Loss Function: Binary Crossentropy
Metrics: Accuracy, Precision, Recall, AUC

📊 Training

Epochs: 70

Batch size: 192

Validation: 20% split

Training and validation curves were plotted to monitor convergence.

✅ Model Evaluation

On the test set, the ANN achieved:

Test loss: 0.373902

Accuracy:  0.854812

Precision: 0.947855

Recall:    0.750833

AUC:       0.900842

F1:        0.837918

(Values may vary slightly depending on random seed and balancing.)

📉 Results & Insights

Balanced performance across Precision, Recall, and F1 indicates the ANN handles class imbalance well.

AUC ~0.90 shows strong discriminatory power between fraud and non-fraud.

Dropout layers helped mitigate overfitting.
