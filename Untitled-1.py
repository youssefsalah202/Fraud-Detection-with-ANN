#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Imports
# -----------------------------
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("TensorFlow:", tf.__version__)

# -----------------------------
# Load & prepare data
# -----------------------------
data = pd.read_csv('fraudTrain.csv')          # read CSV
data = data.select_dtypes(exclude=['object']) # keep only numeric columns

print("Raw shape:", data.shape)
print(data.head())
print(data['is_fraud'].value_counts())

print(data.info())

# Split fraud/non-fraud, then balance
non_fraud = data[data['is_fraud'] == 0]
fraud     = data[data['is_fraud'] == 1]
print("Non-fraud shape:", non_fraud.shape, "| Fraud shape:", fraud.shape)

non_fraud = non_fraud.sample(fraud.shape[0], random_state=0)  # downsample majority
data_bal = pd.concat([fraud, non_fraud], ignore_index=True)
print("Balanced counts:\n", data_bal['is_fraud'].value_counts())

# Features/target
X = data_bal.drop('is_fraud', axis=1)
y = data_bal['is_fraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
print("Shapes:",
      "X_train", X_train.shape,
      "X_test", X_test.shape,
      "y_train", y_train.shape,
      "y_test", y_test.shape)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Convert targets to NumPy
y_train = y_train.to_numpy()
y_test  = y_test.to_numpy()

# -----------------------------
# Helper: prediction for DL models
# -----------------------------
def y_pred_for_DLModels(model, X_test, threshold=0.5):
    y_pred = model.predict(X_test, verbose=0)
    return (y_pred > threshold).flatten()

# -----------------------------
# Build ANN (replacing the CNN)
# -----------------------------
epochs = 70

ANN = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')  # binary classification
])

ANN.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ]
)

# -----------------------------
# Train
# -----------------------------
history = ANN.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=192,
    validation_data=(X_test, y_test),
    verbose=1
)

# -----------------------------
# Plot learning curves
# -----------------------------
def plot_learningCurve(history, epoch):
    epoch_range = range(1, epoch + 1)

    # Accuracy
    plt.figure()
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()

    # Loss
    plt.figure()
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

plot_learningCurve(history, epochs)

# -----------------------------
# Evaluate
# -----------------------------
results = ANN.evaluate(X_test, y_test, verbose=0)
test_loss, accuracy, precision, recall, auc = results
print(f"Test loss: {test_loss:.6f}")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"AUC:       {auc:.6f}")

# F1 score from precision/recall returned by evaluate
F1 = 2 * (precision * recall) / (precision + recall + 1e-8)
print(f"F1:        {F1:.6f}")

# -----------------------------
# Predictions & Confusion Matrix
# -----------------------------
y_pred = y_pred_for_DLModels(ANN, X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.title('Confusion Matrix')
plt.show()
