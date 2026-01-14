# inference_test.py (Automation Ready)
import os
import glob
import numpy as np
import faulthandler
import pandas as pd
import argparse
import sys
# Force matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

faulthandler.enable()

# ============== Argument Parsing ==============
def parse_args():
    parser = argparse.ArgumentParser(description="Automated Inference Test")
    parser.add_argument("--algo", type=str, required=True, choices=['FedAvg', 'FedProx'], help="Algorithm name (case insensitive)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset filename (e.g., V3S1.csv)")
    parser.add_argument("--clients", type=int, required=True, help="Total number of clients")
    parser.add_argument("--thresholds", type=str, required=True, help="Comma-separated thresholds (e.g., 0.05,0.06,0.05)")
    return parser.parse_args()

# ============== Helper Functions ==============
def to_sequence(x, y, seq_size=1):
    x_values, y_values = [], []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

def build_lstm_model(seq_size, n_features):
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
        # Try looking in specific directories if standard path fails
        alt_pattern = f"/app/src/{pattern}"
        weight_files = glob.glob(alt_pattern)
        if not weight_files:
            raise FileNotFoundError(f"No weight files found for {algo_choice}")
    
    def extract_round_number(filename):
        import re
        match = re.search(r'round-(\d+)-weights', filename)
        return int(match.group(1)) if match else 0
    
    latest_weights_file = max(weight_files, key=extract_round_number)
    print(f"Loading weights from: {latest_weights_file}")
    
    weights_npz = np.load(latest_weights_file, allow_pickle=True)
    weights = []
    # Sort keys correctly
    for key in sorted(weights_npz.files, key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
        weights.append(weights_npz[key])
    return weights

def run_inference_for_client(client_id, threshold, model, combined_data, seq_size, algo_choice):
    print(f"Processing Client {client_id} (Threshold: {threshold})")
    
    combined_X, combined_Y = to_sequence(combined_data[['Hz_mod_anomaly']], combined_data['Hz_mod_anomaly'], seq_size)
    combined_predict = model.predict(combined_X, verbose=0)
    combined_mape = np.mean(np.abs(combined_predict - combined_X) / combined_X, axis=1) * 100
    
    # Calculate Metrics
    true_labels = (combined_data['mod_BIN'][seq_size:] != 0).astype(int).values
    predicted_labels = (combined_mape > threshold).astype(int)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Extract TP, FP, TN, FN safely
    TP, FP, TN, FN = 0, 0, 0, 0
    if conf_matrix.shape == (2, 2):
        TN, FP, FN, TP = conf_matrix.ravel()
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    
    return {
        'client_id': client_id, 'threshold': threshold,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision': precision, 'recall': recall, 'f1_score': f1_score, 'accuracy': accuracy
    }

# ============== Main Execution ==============
if __name__ == "__main__":
    args = parse_args()
    
    # Process Thresholds
    try:
        threshold_list = [float(x.strip()) for x in args.thresholds.split(',')]
        if len(threshold_list) != args.clients:
            # If user provided 1 threshold, replicate it for all
            if len(threshold_list) == 1:
                threshold_list = threshold_list * args.clients
            else:
                print(f"Error: Expected {args.clients} thresholds, got {len(threshold_list)}")
                sys.exit(1)
        client_thresholds = {i+1: t for i, t in enumerate(threshold_list)}
    except ValueError:
        print("Error: Thresholds must be comma-separated numbers")
        sys.exit(1)

    # Load Data
    try:
        data_paths = [f'Test_data/{args.dataset}', args.dataset, f'/app/src/Test_data/{args.dataset}']
        data_path = next((p for p in data_paths if os.path.exists(p)), None)
        if not data_path:
            raise FileNotFoundError(f"Dataset {args.dataset} not found")
            
        data1 = pd.read_csv(data_path)
        combined_data = data1.copy()
        combined_data['datetimestamp'] = pd.to_datetime(combined_data['datetimestamp'])
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Load Model
    seq_size = 20
    n_features = 1
    model = build_lstm_model(seq_size, n_features)
    
    try:
        weights = load_latest_weights(args.algo)
        model.set_weights(weights)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    # Run Inference
    all_results = []
    for client_id, threshold in client_thresholds.items():
        res = run_inference_for_client(client_id, threshold, model, combined_data, seq_size, args.algo)
        all_results.append(res)

# Summary & Fairness
    if all_results:
        # 1. Jain's Fairness Index (on F1-Score)
        sum_f1 = sum(r['f1_score'] for r in all_results)
        sum_f1_sq = sum(r['f1_score']**2 for r in all_results)
        N = len(all_results)
        jains = (sum_f1**2) / (N * sum_f1_sq) if sum_f1_sq > 0 else 0
        
        # 2. Average Metrics
        avg_acc = sum(r['accuracy'] for r in all_results) / N
        avg_f1 = sum_f1 / N
        avg_prec = sum(r['precision'] for r in all_results) / N  
        avg_rec = sum(r['recall'] for r in all_results) / N    
        
        print(f"\nFINAL_METRICS_START")
        print(f"JAIN_INDEX:{jains:.4f}")
        print(f"AVG_ACC:{avg_acc:.4f}")
        print(f"AVG_F1:{avg_f1:.4f}")
        print(f"AVG_PREC:{avg_prec:.4f}")  
        print(f"AVG_REC:{avg_rec:.4f}")    
        print(f"FINAL_METRICS_END")

    # Save CSV
    summary_df = pd.DataFrame(all_results)
    csv_name = f'inference_summary_{args.algo}_{args.clients}clients.csv'
    summary_df.to_csv(csv_name, index=False)
    print(f"CSV_GENERATED:{csv_name}")