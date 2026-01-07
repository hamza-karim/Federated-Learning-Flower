# Muhammad Hamza Karim
# Fixed Inference Script (SegFault Safe)

import os
import glob
import numpy as np
import faulthandler
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg") # Force non-interactive backend for Docker stability
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

faulthandler.enable()

# ============== User Inputs ==============
while True:
    algo_choice = input("Select model: 1 for FedAvg, 2 for FedProx: ").lower()
    if algo_choice in ['1', '2', 'fedavg', 'fedprox']:
        if algo_choice == '1':
            algo_choice = 'FedAvg'
        elif algo_choice == '2':
            algo_choice = 'FedProx'
        break
    print("Invalid choice. Please enter 1, 2, 'FedAvg', or 'FedProx'.")

while True:
    dataset_choice = input("Select dataset: 1 for V3S1, 2 for V3S2, 3 for V3S3: ").strip()
    if dataset_choice in ['1', '2', '3']:
        dataset_file = f'V3S{dataset_choice}.csv'
        break
    print("Invalid choice. Please enter 1, 2, or 3.")

while True:
    try:
        total_clients = int(input("Enter total number of clients: "))
        if total_clients <= 0:
            print("Number of clients must be positive.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

# Collect threshold values for each client
client_thresholds = {}
print(f"\nEnter threshold values for {total_clients} clients:")
for client_id in range(1, total_clients + 1):
    while True:
        try:
            threshold = float(input(f"  Client {client_id} threshold: "))
            if threshold <= 0:
                print("  Threshold must be a positive number.")
                continue
            client_thresholds[client_id] = threshold
            break
        except ValueError:
            print("  Invalid input. Please enter a numeric value.")

print(f"\n{'='*50}")
print(f"Configuration:")
print(f"  Algorithm: {algo_choice}")
print(f"  Dataset: {dataset_file}")
print(f"  Total Clients: {total_clients}")
print(f"  Client Thresholds: {client_thresholds}")
print(f"{'='*50}\n")

# ============== Helper Functions ==============
def to_sequence(x, y, seq_size=1):
    """Convert a dataframe into sequences"""
    x_values = []
    y_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

def build_lstm_model(seq_size, n_features):
    """Build and compile LSTM model"""
    model = Sequential()
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                   input_shape=(seq_size, n_features), return_sequences=True))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(RepeatVector(seq_size))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mae', metrics=["mape"])
    return model

def load_latest_weights(algo_choice):
    pattern = f"round-*-weights_{algo_choice.lower()}.npz"
    weight_files = glob.glob(pattern)
    
    if not weight_files:
        print(f"ERROR: No weight files found matching pattern: {pattern}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available .npz files: {glob.glob('*.npz')}")
        raise FileNotFoundError(f"No weight files found for {algo_choice}")
    
    def extract_round_number(filename):
        import re
        match = re.search(r'round-(\d+)-weights', filename)
        return int(match.group(1)) if match else 0
    
    latest_weights_file = max(weight_files, key=extract_round_number)
    round_num = extract_round_number(latest_weights_file)
    print(f"Loading weights from: {latest_weights_file} (Round {round_num})")
    
    weights_npz = np.load(latest_weights_file, allow_pickle=True)
    
    weights = []
    for key in sorted(weights_npz.files, key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
        weights.append(weights_npz[key])
    
    print(f"Loaded {len(weights)} weight arrays")
    return weights

def run_inference_for_client(client_id, threshold, model, combined_data, seq_size):
    """Run inference and generate plots for a specific client"""
    print(f"\n{'='*50}")
    print(f"Processing Client {client_id} (Threshold: {threshold})")
    print(f"{'='*50}")
    
    # Prepare sequences
    combined_X, combined_Y = to_sequence(
        combined_data[['Hz_mod_anomaly']], 
        combined_data['Hz_mod_anomaly'], 
        seq_size
    )
    
    # Run predictions
    combined_predict = model.predict(combined_X, verbose=0)
    combined_mape = np.mean(np.abs(combined_predict - combined_X) / combined_X, axis=1) * 100
    
    # Create results DataFrame
    anomaly_df = pd.DataFrame(combined_data[seq_size:]).copy()
    anomaly_df['combinedMAPE'] = combined_mape
    anomaly_df['max_trainMAPE'] = threshold
    anomaly_df['anomaly'] = anomaly_df['combinedMAPE'] > threshold
    anomaly_df['Hz_mod_anomaly'] = combined_data[seq_size:]['Hz_mod_anomaly'].values
    
    # Filename prefix
    prefix = f"client_{client_id}_{algo_choice}"
    
    # ===== Plot 1: MAPE vs Threshold (FIXED) =====
    plt.figure(figsize=(12, 6))
    if 'datetimestamp' in anomaly_df.columns:
        # FIX: Explicit conversion to numpy array or pydatetime to avoid Pandas/Matplotlib crash
        x_vals = pd.to_datetime(anomaly_df['datetimestamp']).to_numpy()
    else:
        x_vals = np.arange(len(anomaly_df))
        
    plt.plot(x_vals, anomaly_df['combinedMAPE'].values, label='MAPE', linewidth=1.5)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Mean Absolute Percentage Error (MAPE)', fontsize=12)
    plt.title(f'Client {client_id}: Anomaly Detection - MAPE vs Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if 'datetimestamp' in anomaly_df.columns:
        plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'{prefix}_MAPE_vs_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== Plot 2: Anomalies Scatter (FIXED) =====
    combined_anomalies = anomaly_df[anomaly_df['anomaly'] == True]
    plt.figure(figsize=(12, 6))
    
    # Normal data FIX
    if 'datetimestamp' in combined_data.columns:
        x_vals_normal = pd.to_datetime(combined_data['datetimestamp']).to_numpy()
    else:
        x_vals_normal = combined_data.index.to_numpy()
        
    plt.plot(x_vals_normal, combined_data['Hz_mod_anomaly'].values, color='blue', label='Normal Data', linewidth=1, alpha=0.7)
    
    # Anomalies FIX
    if len(combined_anomalies) > 0:
        if 'datetimestamp' in combined_anomalies.columns:
            x_vals_anom = pd.to_datetime(combined_anomalies['datetimestamp']).to_numpy()
        else:
            x_vals_anom = combined_anomalies.index.to_numpy()
            
        plt.scatter(x_vals_anom, combined_anomalies['Hz_mod_anomaly'].values, 
                    color='red', label='Anomalies', s=50, zorder=5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Hz_mod_anomaly', fontsize=12)
    plt.title(f'Client {client_id}: Anomaly Detection on Test Dataset', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'{prefix}_anomalies_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== Confusion Matrix =====
    true_labels = (combined_data['mod_BIN'][seq_size:] != 0).astype(int).values
    predicted_labels = (combined_mape > threshold).astype(int)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title(f'Client {client_id}: Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== Calculate Metrics =====
    TP = conf_matrix[1, 1] if conf_matrix.shape[0] > 1 and conf_matrix.shape[1] > 1 else 0
    FP = conf_matrix[0, 1] if conf_matrix.shape[0] > 0 and conf_matrix.shape[1] > 1 else 0
    TN = conf_matrix[0, 0] if conf_matrix.shape[0] > 0 and conf_matrix.shape[1] > 0 else 0
    FN = conf_matrix[1, 0] if conf_matrix.shape[0] > 1 and conf_matrix.shape[1] > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    
    print(f"\nClient {client_id} Results:")
    print(f"  True Positives (TP): {TP}")
    print(f"  False Positives (FP): {FP}")
    print(f"  True Negatives (TN): {TN}")
    print(f"  False Negatives (FN): {FN}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Total Anomalies Detected: {len(combined_anomalies)}")
    
    return {
        'client_id': client_id,
        'threshold': threshold,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'total_anomalies': len(combined_anomalies)
    }

# ============== Main Processing ==============
# Load the dataset
print(f"Loading test dataset: {dataset_file}...")
data1 = pd.read_csv(f'Test_data/{dataset_file}')
combined_data = data1.copy()
combined_data['datetimestamp'] = pd.to_datetime(combined_data['datetimestamp'])

# Plot the combined dataset (FIXED)
print("Plotting combined dataset...")
plt.figure(figsize=(12, 6))
if 'datetimestamp' in combined_data.columns:
    # FIX: Explicit conversion
    x_vals = pd.to_datetime(combined_data['datetimestamp']).to_numpy()
else:
    x_vals = np.arange(len(combined_data))
    
plt.plot(x_vals, combined_data['Hz_mod_anomaly'].values, linewidth=1)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Hz_mod_anomaly', fontsize=12)
plt.title(f'Test Dataset: {dataset_file} (Pulse + Gaussian)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
if 'datetimestamp' in combined_data.columns:
    plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('combined_dataset_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Model parameters
seq_size = 20
n_features = 1

# Build model
print("\nBuilding LSTM model...")
model = build_lstm_model(seq_size, n_features)

# Load weights
print("\nLoading trained weights...")
try:
    weights = load_latest_weights(algo_choice)
    model.set_weights(weights)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"\nERROR loading weights: {e}")
    print("\nPlease ensure:")
    print("  1. You are in the correct directory containing weight files")
    print("  2. Weight files follow the naming pattern: round-*-weights_FedAvg.npz or round-*-weights_FedProx.npz")
    print("  3. The algorithm choice matches the available weight files")
    exit(1)

# Run inference for each client
all_results = []
for client_id, threshold in client_thresholds.items():
    try:
        result = run_inference_for_client(client_id, threshold, model, combined_data, seq_size)
        all_results.append(result)
    except Exception as e:
        print(f"\nERROR processing Client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============== Summary Report ==============
print(f"\n{'='*70}")
print(f"SUMMARY REPORT - {algo_choice}")
print(f"{'='*70}")
print(f"{'Client':<8} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
print(f"{'-'*70}")
for result in all_results:
    print(f"{result['client_id']:<8} {result['threshold']:<12.4f} {result['precision']:<12.4f} "
          f"{result['recall']:<12.4f} {result['f1_score']:<12.4f} {result['accuracy']:<12.4f}")
print(f"{'='*70}")

# Calculate and display average metrics
if all_results:
    avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
    avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
    avg_f1_score = sum(r['f1_score'] for r in all_results) / len(all_results)
    avg_accuracy = sum(r['accuracy'] for r in all_results) / len(all_results)
    
    print(f"{'AVERAGE':<8} {'-':<12} {avg_precision:<12.4f} "
          f"{avg_recall:<12.4f} {avg_f1_score:<12.4f} {avg_accuracy:<12.4f}")
    print(f"{'='*70}")

# Save summary to CSV
summary_df = pd.DataFrame(all_results)
summary_filename = f'inference_summary_{algo_choice}_{total_clients}clients.csv'
summary_df.to_csv(summary_filename, index=False)
print(f"\nSummary saved to: {summary_filename}")
print("\nInference testing completed successfully!")