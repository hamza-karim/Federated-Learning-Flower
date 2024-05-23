import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the model architecture
seq_size = 20
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(seq_size, 1), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(seq_size))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))

# Load weights
weights = np.load('round-7-weights.npz')
model.set_weights([weights[key] for key in weights])

# Load the first dataset
data1 = pd.read_csv('dataset/V2_25_Feb_3_60m_anomalous_data(pulse).csv')

# Load the second dataset
data2 = pd.read_csv('dataset/V1_25_Feb_180m_anomalous_data(gaussian).csv')

# Combine the datasets
combined_data = pd.concat([data1, data2], ignore_index=True)

# Preprocess the combined dataset
combined_data['datetimestamp'] = pd.to_datetime(combined_data['datetimestamp'])

# Plot the combined dataset
plt.figure(figsize=(10, 6))
sns.lineplot(x='datetimestamp', y='Hz_mod_anomaly', data=combined_data)
plt.xticks(rotation=45)
plt.gcf().set_facecolor('white')
plt.xlabel('Date')
plt.ylabel('Hz_mod_anomaly')
plt.title('Combined Dataset (Pulse + Gaussian)')
plt.savefig('/app/src/combined_dataset.png')
plt.close()

# Convert the combined dataset into sequences
def to_sequence(x, y, seq_size=1):
    x_values = []
    y_values = []
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
    return np.array(x_values), np.array(y_values)

combined_X, combined_Y = to_sequence(combined_data[['Hz_mod_anomaly']], combined_data['Hz_mod_anomaly'], seq_size)

# Use the trained model to predict the reconstruction errors (MAPE) on the combined dataset
combined_predict = model.predict(combined_X)
combined_mape = np.mean(np.abs(combined_predict - combined_X) / combined_X, axis=1) * 100

# Thresholding using MAPE
max_trainMAPE = 0.85

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
plt.savefig('/app/src/anomaly_detection.png')
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
plt.savefig('/app/src/anomalies.png')
plt.close()

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
plt.savefig('/app/src/confusion_matrix.png')
plt.close()

# Initialize counts for metrics
TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

# Calculate total number of normal and anomalous instances
total_normal = len(combined_data[combined_data['mod_BIN'] == 0]) - seq_size
total_anomalous = len(combined_data[combined_data['mod_BIN'] != 0]) - seq_size

# Calculate percentages
TP_percentage = (TP / total_anomalous) * 100
FP_percentage = (FP / total_normal) * 100
TN_percentage = (TN / total_normal) * 100
FN_percentage = (FN / total_anomalous) * 100

# Print the counts and percentages
print("True Positives (Anomalies correctly predicted as anomalies):", TP, f"({TP_percentage:.2f}%)")
print("False Positives (Normal data incorrectly predicted as anomalies):", FP, f"({FP_percentage:.2f}%)")
print("True Negatives (Normal data correctly predicted as normal):", TN, f"({TN_percentage:.2f}%)")
print("False Negatives (Anomalies incorrectly predicted as normal):", FN, f"({FN_percentage:.2f}%)")

# Calculate Precision, Recall, and F1-score
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
