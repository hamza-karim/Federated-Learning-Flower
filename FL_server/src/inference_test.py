# inference_test.py (MAE-based with Fixed Training Scaling)
import os
import glob
import numpy as np
import faulthandler
import pandas as pd
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
# MinMaxScaler import kept for compatibility, but we use manual scaling
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from keras.optimizers import Adam

faulthandler.enable()

# ============== Argument Parsing ==============
def parse_args():
    parser = argparse.ArgumentParser(description="Automated Inference Test")
    parser.add_argument("--algo", type=str, required=True, choices=['FedAvg', 'FedProx'], help="Algorithm name (case insensitive)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset filename (e.g., V3S1.csv)")
    parser.add_argument("--clients", type=int, required=True, help="Total number of clients")
    parser.add_argument("--thresholds", type=str, required=True, help="Comma-separated thresholds (e.g., 0.009238,0.009238)")
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
    model.add(LSTM(256, activation='tanh', recurrent_activation='sigmoid',
                   input_shape=(seq_size, n_features), return_sequences=True))
    model.add(Dropout(0.1419))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(Dropout(0.1419))
    model.add(RepeatVector(seq_size))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.1419))
    model.add(LSTM(256, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(Dropout(0.1419))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(
        optimizer=Adam(learning_rate=0.000549), 
        loss='mae',
        metrics=['mape']
    )
    return model

def load_latest_weights(algo_choice):
    pattern = f"round-*-weights_{algo_choice.lower()}.npz"
    weight_files = glob.glob(pattern)
    
    if not weight_files:
        print(f"ERROR: No weight files found matching pattern: {pattern}")
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
    for key in sorted(weights_npz.files, key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
        weights.append(weights_npz[key])
    return weights

def run_inference_for_client(client_id, threshold, model, combined_data, seq_size, algo_choice):
    """
    Run inference using MAE-based reconstruction error with FIXED Training Scaling.
    """
    print(f"Processing Client {client_id} (Threshold: {threshold:.6f})...", end=" ")
    
    # ============================================
    # FIXED SCALING: Using Training Bounds (From Trial 39)
    # ============================================
    TRAIN_MIN = 59.93999863
    TRAIN_MAX = 60.04000092
    
    # Extract raw Hz values
    hz_values = combined_data[['Hz_mod_anomaly']].values
    
    # Apply Manual MinMax Scaling
    hz_scaled = (hz_values - TRAIN_MIN) / (TRAIN_MAX - TRAIN_MIN)
    scaled_df = pd.DataFrame(hz_scaled, columns=['Hz_scaled'])
    
    # Create sequences from SCALED data
    combined_X, combined_Y = to_sequence(
        scaled_df[['Hz_scaled']], 
        scaled_df['Hz_scaled'], 
        seq_size
    )
    
    # Predict (reconstruct)
    combined_predict = model.predict(combined_X, verbose=0)
    
    # ============================================
    # Calculate MAE
    # ============================================
    combined_mae = np.mean(
        np.abs(combined_predict - combined_X),
        axis=(1, 2)
    )
    
    # Calculate Metrics
    true_labels = (combined_data['mod_BIN'][seq_size:] != 0).astype(int).values
    predicted_labels = (combined_mae > threshold).astype(int)
    
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Extract TP, FP, TN, FN safely
    TP, FP, TN, FN = 0, 0, 0, 0
    if conf_matrix.shape == (2, 2):
        TN, FP, FN, TP = conf_matrix.ravel()
    elif conf_matrix.shape == (1, 1):
        if true_labels[0] == 0: TN = conf_matrix[0,0]
        else: TP = conf_matrix[0,0]
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    
    # Concise logging
    print(f"Done. Acc: {accuracy:.4f} | F1: {f1_score:.4f} | TP: {TP} | FP: {FP}")
    
    return {
        'client_id': client_id, 
        'threshold': threshold,
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision': precision, 
        'recall': recall, 
        'f1_score': f1_score, 
        'accuracy': accuracy,
        'mae_mean': float(combined_mae.mean()),
        'mae_median': float(np.median(combined_mae))
    }

# ============== Main Execution ==============
if __name__ == "__main__":
    args = parse_args()
    
    print("="*80)
    print(f" FEDERATED LEARNING INFERENCE TEST | {args.algo} | {args.dataset}")
    print("="*80)
    
    # Process Thresholds
    try:
        threshold_list = [float(x.strip()) for x in args.thresholds.split(',')]
        if len(threshold_list) != args.clients:
            if len(threshold_list) == 1:
                print(f"Applying threshold {threshold_list[0]:.6f} to all {args.clients} clients")
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
        data_paths = [
            f'Test_data/{args.dataset}', 
            args.dataset, 
            f'/app/src/Test_data/{args.dataset}'
        ]
        data_path = next((p for p in data_paths if os.path.exists(p)), None)
        if not data_path:
            raise FileNotFoundError(f"Dataset {args.dataset} not found")
        
        print(f"Loading dataset: {data_path} ...", end=" ")
        data1 = pd.read_csv(data_path)
        combined_data = data1.copy()
        combined_data['datetimestamp'] = pd.to_datetime(combined_data['datetimestamp'])
        print(f"Loaded {combined_data.shape[0]} rows.")
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        sys.exit(1)

    # Load Model
    seq_size = 20
    n_features = 1
    
    print(f"Building LSTM Model (Seq: {seq_size})...", end=" ")
    model = build_lstm_model(seq_size, n_features)
    print("Done.")
    
    try:
        print(f"Loading weights...", end=" ")
        weights = load_latest_weights(args.algo)
        model.set_weights(weights)
        print(f"Success.")
    except Exception as e:
        print(f"\n❌ Error loading weights: {e}")
        sys.exit(1)

    # Run Inference
    print("-" * 80)
    
    all_results = []
    for client_id, threshold in client_thresholds.items():
        res = run_inference_for_client(
            client_id, threshold, model, combined_data, seq_size, args.algo
        )
        all_results.append(res)

    # Summary & Fairness
    if all_results:
        # Calculate Metrics manually to ensure correctness
        sum_f1 = sum(r['f1_score'] for r in all_results)
        sum_f1_sq = sum(r['f1_score']**2 for r in all_results)
        N = len(all_results)
        
        # Jain's Fairness Index
        jains = (sum_f1**2) / (N * sum_f1_sq) if sum_f1_sq > 0 else 0
        
        # Direct Averages (Correct way)
        avg_acc = sum(r['accuracy'] for r in all_results) / N
        avg_f1 = sum_f1 / N
        avg_prec = sum(r['precision'] for r in all_results) / N  
        avg_rec = sum(r['recall'] for r in all_results) / N
        
        # Calculate MIN F1 (Worst Case Performance)
        min_f1 = min(r['f1_score'] for r in all_results)
        
        # Pretty Print Summary Table
        print("\n" + "="*80)
        print(f"{'AGGREGATE PERFORMANCE SUMMARY':^80}")
        print("="*80)
        print(f"  Total Clients         :  {N}")
        print("-" * 40)
        print(f"  Jain's Fairness Index :  {jains:.4f}")
        print(f"  Average Accuracy      :  {avg_acc:.4f}")
        print(f"  Average Precision     :  {avg_prec:.4f}")
        print(f"  Average Recall        :  {avg_rec:.4f}")
        print(f"  Average F1-Score      :  {avg_f1:.4f}")
        print("-" * 40)
        print(f"  WORST CLIENT F1       :  {min_f1:.4f}")
        print("="*80)
        
        # Machine-readable output for Dashboard Parsing
        print(f"\nFINAL_METRICS_START")
        print(f"JAIN_INDEX:{jains:.4f}")
        print(f"AVG_ACC:{avg_acc:.4f}")
        print(f"AVG_F1:{avg_f1:.4f}")
        print(f"AVG_PREC:{avg_prec:.4f}")   
        print(f"AVG_REC:{avg_rec:.4f}")     
        print(f"MIN_F1:{min_f1:.4f}")
        print(f"FINAL_METRICS_END")

    # Save CSV
    summary_df = pd.DataFrame(all_results)
    dataset_name = os.path.splitext(args.dataset)[0]
    csv_name = f'inference_summary_{args.algo}_{dataset_name}_{args.clients}clients.csv'
    
    summary_df.to_csv(csv_name, index=False)
    print(f"\n✅ CSV Saved: {csv_name}")
    print(f"CSV_GENERATED:{csv_name}")