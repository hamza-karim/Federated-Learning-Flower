# Muhammad Hamza Karim

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to convert a dataframe into sequences
def to_sequence(x, y, seq_size=1):
    x_values = []
    y_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

# Load the trained model
model = joblib.load('trained_model_LSTM.joblib')

# Load the dataset
data1 = pd.read_csv('dataset/V2_25_Feb_3_60m_anomalous_data(pulse).csv')
data2 = pd.read_csv('dataset/V1_25_Feb_180m_anomalous_data(gaussian).csv')

# Combined
combined_data = pd.concat([data1, data2], ignore_index=True)
combined_data['datetimestamp'] = pd.to_datetime(combined_data['datetimestamp'])

# Plot the combined dataset
plt.figure(figsize=(10, 6))
sns.lineplot(x='datetimestamp', y='Hz_mod_anomaly', data=combined_data)
plt.xticks(rotation=45)
plt.gcf().set_facecolor('white')
plt.xlabel('Date')
plt.ylabel('Hz_mod_anomaly')
plt.title('Combined Dataset (Pulse + Gaussian)')
plt.savefig('combined_dataset_plot.png')
plt.close()

seq_size = 20  # Use the same sequence size as used in training
combined_X, combined_Y = to_sequence(combined_data[['Hz_mod_anomaly']], combined_data['Hz_mod_anomaly'], seq_size)

# Use the trained model to predict the reconstruction errors (MAPE) on the combined dataset
combined_predict = model.predict(combined_X)
combined_mape = np.mean(np.abs(combined_predict - combined_X) / combined_X, axis=1) * 100

# Thresholding using MAPE
max_trainMAPE = 42.8

# Capture all details in a DataFrame for easy plotting
anomaly_df = pd.DataFrame(combined_data[seq_size:])
anomaly_df['combinedMAPE'] = combined_mape
anomaly_df['max_trainMAPE'] = max_trainMAPE
anomaly_df['anomaly'] = anomaly_df['combinedMAPE'] > max_trainMAPE
anomaly_df['Hz_mod_anomaly'] = combined_data[seq_size:]['Hz_mod_anomaly']

# Plot combined MAPE vs max_trainMAPE
plt.figure(figsize=(10, 6))
sns.lineplot(x=anomaly_df.index, y=anomaly_df['combinedMAPE'], label='MAPE')
sns.lineplot(x=anomaly_df.index, y=anomaly_df['max_trainMAPE'], label='Threshold', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.title('Anomaly Detection on Combined Dataset (Pulse + Gaussian)')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('combined_MAPE_vs_threshold.png')
plt.close()

# Identify anomalies based on the threshold
combined_anomalies = anomaly_df[anomaly_df['anomaly'] == True]

# Plot anomalies
plt.figure(figsize=(10, 6))
sns.lineplot(x=combined_data['datetimestamp'], y=combined_data['Hz_mod_anomaly'], color='blue', label='Normal Data')
sns.scatterplot(x=combined_anomalies['datetimestamp'], y=combined_anomalies['Hz_mod_anomaly'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Hz_mod_anomaly')
plt.title('Anomaly Detection on New Combined Dataset (Pulse + Gaussian)')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('combined_anomalies_plot.png')
plt.close()

# Calculate and plot the confusion matrix
# Get the true labels
true_labels = (combined_data['mod_BIN'][seq_size:] != 0).astype(int)

# Create the confusion matrix
conf_matrix = confusion_matrix(true_labels, combined_mape > max_trainMAPE)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('combined_confusion_matrix.png')
plt.close()

# Initialize counts for metrics
TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

print("True Positives (Anomalies correctly predicted as anomalies):", TP)
print("False Positives (Normal data incorrectly predicted as anomalies):", FP)
print("True Negatives (Normal data correctly predicted as normal):", TN)
print("False Negatives (Anomalies incorrectly predicted as normal):", FN)

# Calculate Precision, Recall, and F1-score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
