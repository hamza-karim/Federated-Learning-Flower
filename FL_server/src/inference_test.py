# Muhammad Hamza Karim

import os
import glob
import numpy as np
import faulthandler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

faulthandler.enable()

while True:
    algo_choice = input("Select model: 1 for FedAvg, 2 for FedProx: ").lower()
    if algo_choice in ['1', '2', 'FedAvg', 'FedProx']:
        if algo_choice == '1':
            algo_choice = 'FedAvg'
        elif algo_choice == '2':
            algo_choice = 'FedProx'
        break
    print("Invalid choice. Please enter 1, 2, 'FedAvg', or 'FedProx'.")

while True:
    try:
        max_trainMAPE = float(input("Enter threshold value:"))
        if max_trainMAPE <= 0:
            print("Threshold must be a positive number.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

# Function to convert a dataframe into sequences
def to_sequence(x, y, seq_size=1):
    x_values = []
    y_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

# Load the dataset
data1 = pd.read_csv('Test_data/V3S1.csv')
combined_data = data1
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

#=== Rebuild the model ===
n_features = 1  # Hz_mod_anomaly is the only feature
model = Sequential()
model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
               input_shape=(seq_size, n_features), return_sequences=True))
model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
model.add(RepeatVector(seq_size))
model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mae', metrics=["mape"])


# === Load latest round weights ===
from flwr.common import Parameters, parameters_to_ndarrays
pattern = f"round-*-weights_{algo_choice}.npz"  # FedAvg or FedProx
weight_files = sorted(glob.glob(pattern), key=os.path.getmtime)
latest_weights_file = weight_files[-1]
print(f"Loading latest weights from: {latest_weights_file}")

# Load npz file with pickled Parameters object
weights_npz = np.load(latest_weights_file, allow_pickle=True)

# Extract the first stored object (Parameters)
param_obj = weights_npz["arr_0"].item()
print("Type of param_obj:", type(param_obj))
# Convert to list of ndarrays
if isinstance(param_obj, Parameters):
    weights = parameters_to_ndarrays(param_obj)
else:
    raise ValueError("Expected fl.common.Parameters, but got:", type(param_obj))

# Print shape info
print("Shapes of weights loaded:")
for i, w in enumerate(weights):
    print(f"Weight {i}: {w.shape}")

# Set weights to model
model.set_weights(weights)

# === Run predictions and anomaly detection ===
combined_predict = model.predict(combined_X)
combined_mape = np.mean(np.abs(combined_predict - combined_X) / combined_X, axis=1) * 100

# Results DataFrame
anomaly_df = pd.DataFrame(combined_data[seq_size:]).copy()
anomaly_df['combinedMAPE'] = combined_mape
anomaly_df['max_trainMAPE'] = max_trainMAPE
anomaly_df['anomaly'] = anomaly_df['combinedMAPE'] > max_trainMAPE
anomaly_df['Hz_mod_anomaly'] = combined_data[seq_size:]['Hz_mod_anomaly']

# MAPE Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=anomaly_df.index, y=anomaly_df['combinedMAPE'], label='MAPE')
sns.lineplot(x=anomaly_df.index, y=anomaly_df['max_trainMAPE'], label='Threshold', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.title('Anomaly Detection on Test Dataset')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('combined_MAPE_vs_threshold.png')
plt.close()

# Anomaly Plot
combined_anomalies = anomaly_df[anomaly_df['anomaly'] == True]
plt.figure(figsize=(10, 6))
sns.lineplot(x=combined_data['datetimestamp'], y=combined_data['Hz_mod_anomaly'], color='blue', label='Normal Data')
sns.scatterplot(x=combined_anomalies['datetimestamp'], y=combined_anomalies['Hz_mod_anomaly'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Hz_mod_anomaly')
plt.title('Anomaly Detection on New Test Dataset')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('combined_anomalies_plot.png')
plt.close()

# Confusion Matrix
true_labels = (combined_data['mod_BIN'][seq_size:] != 0).astype(int)
conf_matrix = confusion_matrix(true_labels, combined_mape > max_trainMAPE)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('combined_confusion_matrix.png')
plt.close()

# Metrics Calculation
TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

print("True Positives (Anomalies correctly predicted as anomalies):", TP)
print("False Positives (Normal data incorrectly predicted as anomalies):", FP)
print("True Negatives (Normal data correctly predicted as normal):", TN)
print("False Negatives (Anomalies incorrectly predicted as normal):", FN)

precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
accuracy = (TP + TN) / (TP + FP + TN + FN)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("Accuracy:", accuracy)
