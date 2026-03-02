# Muhammad Hamza Karim
# FINAL CLIENT CODE - MAE-based Reconstruction Error

import os
# Force CPU/GPU threading limits to prevent resource fighting
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "default"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

import joblib
import json
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import random
import time

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth on {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
        
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

import faulthandler
import warnings
from keras import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter('ignore')
faulthandler.enable()

# ---------------- Environment Variables ---------------- #
SERVER_IP = os.getenv("SERVER_IP", "10.226.47.97")
SERVER_PORT = os.getenv("SERVER_PORT", "8080")
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))  
TOTAL_CLIENTS = int(os.getenv("TOTAL_CLIENTS", "20"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
MODEL = os.getenv("MODEL", "lstm")
ALGO = os.getenv("ALGO", "fedavg")
DISABLE_PLOTS = os.getenv("DISABLE_PLOTS", "false").lower() == "true"
THRESHOLD_PERCENTILE = float(os.getenv("THRESHOLD_PERCENTILE", "99.6"))

SERVER_ADDR = f"{SERVER_IP}:{SERVER_PORT}"
temp_loss = []
temp_mape = []

# Set random seed
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ---------------- Safe Plotting Wrapper ---------------- #
def safe_plot(plot_func, filename, *args, **kwargs):
    """Wrapper to safely execute plotting with proper error handling"""
    if DISABLE_PLOTS:
        print(f"Plotting disabled. Skipping: {filename}")
        return
    
    try:
        plot_func(*args, **kwargs)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {filename}")
    except Exception as e:
        print(f"✗ Plot failed ({filename}): {e}")
    finally:
        plt.close('all')  # Ensure all figures are closed

# ---------------- Dataset Loading ---------------- #
def load_dataset():
    """Load preprocessed active dataset (V_0_processed_active.csv)"""
    folder_path = "./Train_data"
    target_file = "V_0_processed_active.csv"
    file_path = os.path.join(folder_path, target_file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{target_file} not found in {folder_path}")
    
    print(f"Loading dataset: {target_file}")
    dataframe = pd.read_csv(file_path)
    dataframe["datetimeCST"] = pd.to_datetime(dataframe["datetimeCST"], errors="coerce")

    df = dataframe[["datetimeCST", "Hz"]].copy()
    df.set_index("datetimeCST", inplace=True)
    
    print(f"Dataset loaded: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    def plot_frequency_week():
        x_vals = df.index.to_pydatetime()
        y_vals = df["Hz"].values
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, y_vals, linewidth=1, color='green', alpha=0.7)
        plt.title("Frequency Variation (Hz) Over One Week", fontsize=14, fontweight="bold")
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.gcf().autofmt_xdate()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

    safe_plot(plot_frequency_week, "frequency_week_plot.png")
    return df

# ---------------- Dataset Preprocessing ---------------- #
def preprocess_dataset(
    df: pd.DataFrame,
    total_clients: int,
    client_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess dataset with MinMax scaling and sequence creation"""
    
    normal_data = df.copy()

    # =====================================================
    # 1. MIN-MAX SCALING
    # =====================================================
    scaler = MinMaxScaler()
    hz_values = normal_data[["Hz"]].values
    hz_scaled = scaler.fit_transform(hz_values)
    normal_data["Hz_scaled"] = hz_scaled

    print("\nMinMax Scaling Applied")
    print(f"   Scaled shape: {hz_scaled.shape}")
    print(f"   Min: {hz_scaled.min():.4f} | Max: {hz_scaled.max():.4f}")

    # =====================================================
    # 2. PLOT FULL SCALED DATA
    # =====================================================
    def plot_scaled_full():
        x_vals = normal_data.index.to_pydatetime()
        y_vals = normal_data["Hz_scaled"].values
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, y_vals, linewidth=1, color='blue', alpha=0.7)
        plt.title("Scaled Frequency (MinMax) Over Full Dataset", fontsize=14, fontweight="bold")
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("Scaled Frequency (0–1)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

    safe_plot(plot_scaled_full, "frequency_scaled_full.png")

    # =====================================================
    # 3. DATASET SPLITTING
    # =====================================================
    total_rows = len(normal_data)
    rows_per_client = total_rows // total_clients

    print(f"\n{'='*60}")
    print(f"Dataset Distribution Summary")
    print(f"{'='*60}")
    print(f"Total dataset size: {total_rows:,} rows")
    print(f"Rows per client: {rows_per_client:,} rows")
    print(f"Total clients: {total_clients}")
    print(f"{'='*60}\n")

    # =====================================================
    # 4. PLOT ALL CLIENTS
    # =====================================================
    def plot_all_clients():
        plt.figure(figsize=(12, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, total_clients))
        for i in range(total_clients):
            start_idx = i * rows_per_client
            end_idx = start_idx + rows_per_client if i != total_clients - 1 else total_rows
            client_slice = normal_data.iloc[start_idx:end_idx]
            x_vals = client_slice.index.to_pydatetime()
            y_vals = client_slice["Hz"].values
            plt.plot(x_vals, y_vals, color=colors[i], label=f"Client {i+1}", linewidth=1)

        plt.title(f"Frequency Variation (Hz) Divided for {total_clients} Clients", 
                  fontsize=14, fontweight="bold")
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

    safe_plot(plot_all_clients, f"frequency_divided_{total_clients}_clients.png")

    # =====================================================
    # 5. CLIENT SLICE
    # =====================================================
    start_idx = (client_id - 1) * rows_per_client
    end_idx = start_idx + rows_per_client if client_id != total_clients else total_rows
    client_data = normal_data.iloc[start_idx:end_idx]

    print(f"Client {client_id} Data Allocation:")
    print(f"  - Total rows: {len(client_data):,}")
    print(f"  - Start index: {start_idx:,}")
    print(f"  - End index: {end_idx:,}")
    print(f"  - Percentage: {(len(client_data)/total_rows)*100:.2f}%")

    # =====================================================
    # 6. CLIENT PLOT
    # =====================================================
    def plot_client_data():
        x_vals = client_data.index.to_pydatetime()
        y_vals = client_data["Hz"].values
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, y_vals, color="teal", linewidth=1)
        plt.title(f"Frequency Variation (Hz) for Client {client_id}\n"
                  f"Dataset Size: {len(client_data):,} rows", 
                  fontsize=14, fontweight="bold")
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

    safe_plot(plot_client_data, f"client_{client_id}_frequency_plot.png")

    # =====================================================
    # 7. TRAIN / TEST SPLIT
    # =====================================================
    train_size = int(0.9 * len(client_data))
    train = client_data.iloc[:train_size]
    test = client_data.iloc[train_size:]

    print(f"  - Train rows: {len(train):,} (90%)")
    print(f"  - Test rows : {len(test):,} (10%)\n")

    # =====================================================
    # 8. SEQUENCE CREATION
    # =====================================================
    seq_size = 20
    
    def to_sequence(x, y, seq_size=1):
        x_vals, y_vals = [], []
        for i in range(len(x) - seq_size):
            x_vals.append(x.iloc[i:(i + seq_size)].values)
            y_vals.append(y.iloc[i + seq_size])
        return np.array(x_vals), np.array(y_vals)

    X_train, y_train = to_sequence(train[["Hz_scaled"]], train["Hz_scaled"], seq_size)
    X_test, y_test = to_sequence(test[["Hz_scaled"]], test["Hz_scaled"], seq_size)

    print(f"Sequence Data Shapes (seq_size={seq_size}):")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - X_test : {X_test.shape}")
    print(f"  - y_test : {y_test.shape}")
    print(f"{'='*60}\n")

    return X_train, y_train, X_test, y_test

# ---------------- Model Builders ---------------- #
# def build_lstm(input_shape):
#     """Build LSTM Autoencoder with MAE loss"""
#     model = Sequential()
#     model.add(LSTM(256, activation='tanh', recurrent_activation='sigmoid', 
#                    input_shape=input_shape, return_sequences=True))
#     model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', 
#                    return_sequences=False))
#     model.add(RepeatVector(input_shape[0]))
#     model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', 
#                    return_sequences=True))
#     model.add(LSTM(256, activation='tanh', recurrent_activation='sigmoid', 
#                    return_sequences=True))
#     model.add(TimeDistributed(Dense(input_shape[1])))
    
#     # Compile with MAE loss and MAPE metric
#     model.compile(optimizer='adam', loss='mae', metrics=['mape'])
#     return model

# ---------------- Model Builders ---------------- #
def build_lstm(input_shape):
    """Build LSTM Autoencoder with MAE loss (Trial 39 config)"""

    model = Sequential()

    # -------- Encoder --------
    model.add(LSTM(
        256,
        activation='tanh',
        recurrent_activation='sigmoid',
        input_shape=input_shape,
        return_sequences=True
    ))
    model.add(Dropout(0.1419))

    model.add(LSTM(
        64,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=False
    ))
    model.add(Dropout(0.1419))

    # -------- Bottleneck --------
    model.add(RepeatVector(input_shape[0]))

    # -------- Decoder --------
    model.add(LSTM(
        64,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=True
    ))
    model.add(Dropout(0.1419))

    model.add(LSTM(
        256,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=True
    ))
    model.add(Dropout(0.1419))

    model.add(TimeDistributed(Dense(input_shape[1])))

    # -------- Adam with Trial 39 LR --------
    optimizer = Adam(learning_rate=0.000549)

    # -------- Compile --------
    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mape']
    )

    return model

def build_bilstm(input_shape):
    """Build Bidirectional LSTM Autoencoder with MAE loss"""
    model = Sequential()
    model.add(Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid', 
                                 return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', 
                                 return_sequences=False)))
    model.add(RepeatVector(input_shape[0]))
    model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', 
                                 return_sequences=True)))
    model.add(Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid', 
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(input_shape[1])))
    
    # Compile with MAE loss and MAPE metric
    model.compile(optimizer='adam', loss='mae', metrics=['mape'])
    return model

# ---------------- Flower Clients ---------------- #
# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
#         self.model = model
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
#         self.client_id = client_id 

#     def get_properties(self, config):
#         return {"client_id": int(self.client_id)}

#     def get_parameters(self, config):
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         history = self.model.fit(
#             self.X_train, 
#             self.X_train,  # Autoencoder: input = output
#             epochs=EPOCHS, 
#             batch_size=64, 
#             validation_split=0.2, 
#             verbose=1
#         )
#         final_train_loss = history.history['loss'][-1]
#         return self.model.get_weights(), len(self.X_train), {"train_loss": final_train_loss}

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         loss, mape = self.model.evaluate(self.X_test, self.X_test, verbose=0)
#         temp_loss.append(loss)
#         temp_mape.append(mape)
#         print(f"Eval Loss (MAE): {loss:.6f} || Eval MAPE: {mape:.4f}")
#         return loss, len(self.X_test), {"mape": mape}


# class FedProxClient(fl.client.NumPyClient):
#     def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
#         self.model = model
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
#         self.client_id = client_id 
#         self.global_weights = None
#         self.mu = 0.0

#     def get_properties(self, config):
#         return {"client_id": int(self.client_id)}

#     def get_parameters(self, config):
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         self.global_weights = parameters
#         self.mu = config.get("proximal_mu", 0.0)
#         self.model.compile(optimizer="adam", loss=self.fedprox_loss)
#         history = self.model.fit(
#             self.X_train, 
#             self.X_train,  # Autoencoder: input = output
#             epochs=EPOCHS, 
#             batch_size=32, 
#             validation_split=0.2, 
#             verbose=1
#         )
#         final_train_loss = history.history['loss'][-1]
#         return self.model.get_weights(), len(self.X_train), {"train_loss": final_train_loss}

#     def fedprox_loss(self, y_true, y_pred):
#         base_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
#         prox_term = 0.0
#         for w, w0 in zip(self.model.trainable_weights, self.global_weights):
#             prox_term += tf.reduce_sum(tf.square(w - tf.convert_to_tensor(w0)))
#         return base_loss + (self.mu / 2.0) * prox_term

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         self.model.compile(optimizer="adam", loss="mae", metrics=["mape"])
#         loss, mape = self.model.evaluate(self.X_test, self.X_test, verbose=0)
#         temp_loss.append(loss)
#         temp_mape.append(mape)
#         print(f"Eval Loss (MAE): {loss:.6f} || Eval MAPE: {mape:.4f}")
#         return loss, len(self.X_test), {"mape": mape}

# ---------------- Flower Clients ---------------- #
class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id 

    def get_properties(self, config):
        return {"client_id": int(self.client_id)}

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # ============================================
        # CHANGE 1: Record training start time
        # ============================================
        fit_start_time = time.time()
        
        history = self.model.fit(
            self.X_train, 
            self.X_train,  # Autoencoder: input = output
            epochs=EPOCHS, 
            batch_size=64, 
            validation_split=0.2, 
            verbose=1
        )
        
        # ============================================
        # CHANGE 2: Calculate training duration
        # ============================================
        fit_end_time = time.time()
        training_duration = fit_end_time - fit_start_time
        
        final_train_loss = history.history['loss'][-1]
        
        # ============================================
        # CHANGE 3: Return client_id and duration in metrics
        # ============================================
        return self.model.get_weights(), len(self.X_train), {
            "train_loss": final_train_loss,
            "client_id": int(self.client_id),          
            "training_duration": float(training_duration)  
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mape = self.model.evaluate(self.X_test, self.X_test, verbose=0)
        temp_loss.append(loss)
        temp_mape.append(mape)
        print(f"Eval Loss (MAE): {loss:.6f} || Eval MAPE: {mape:.4f}")
        return loss, len(self.X_test), {"mape": mape}


class FedProxClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id 
        self.global_weights = None
        self.mu = 0.0

    def get_properties(self, config):
        return {"client_id": int(self.client_id)}

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.global_weights = parameters
        self.mu = config.get("proximal_mu", 0.0)
        self.model.compile(optimizer="adam", loss=self.fedprox_loss)
        
        # ============================================
        # CHANGE 1: Record training start time
        # ============================================
        fit_start_time = time.time()
        
        history = self.model.fit(
            self.X_train, 
            self.X_train,  # Autoencoder: input = output
            epochs=EPOCHS, 
            batch_size=32, 
            validation_split=0.2, 
            verbose=1
        )
        
        # ============================================
        # CHANGE 2: Calculate training duration
        # ============================================
        fit_end_time = time.time()
        training_duration = fit_end_time - fit_start_time
        
        final_train_loss = history.history['loss'][-1]
        
        # ============================================
        # CHANGE 3: Return client_id and duration in metrics
        # ============================================
        return self.model.get_weights(), len(self.X_train), {
            "train_loss": final_train_loss,
            "client_id": int(self.client_id),          
            "training_duration": float(training_duration) 
        }

    def fedprox_loss(self, y_true, y_pred):
        base_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        prox_term = 0.0
        for w, w0 in zip(self.model.trainable_weights, self.global_weights):
            prox_term += tf.reduce_sum(tf.square(w - tf.convert_to_tensor(w0)))
        return base_loss + (self.mu / 2.0) * prox_term

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer="adam", loss="mae", metrics=["mape"])
        loss, mape = self.model.evaluate(self.X_test, self.X_test, verbose=0)
        temp_loss.append(loss)
        temp_mape.append(mape)
        print(f"Eval Loss (MAE): {loss:.6f} || Eval MAPE: {mape:.4f}")
        return loss, len(self.X_test), {"mape": mape}


# ---------------- Main ---------------- #
if __name__ == "__main__":
    print(f"\n===== Client {CLIENT_ID} Configuration =====")
    print(f"Server Address : {SERVER_ADDR}")
    print(f"Total Clients  : {TOTAL_CLIENTS}")
    print(f"Model Selected : {MODEL}")
    print(f"Iteration Per Round : {EPOCHS}")
    print(f"Algorithm      : {ALGO.capitalize()}")
    print(f"Plotting       : {'Disabled' if DISABLE_PLOTS else 'Enabled'}")
    print(f"Threshold Percentile : {THRESHOLD_PERCENTILE}%")
    print(f"=============================================")

    df = load_dataset()
    X_train, y_train, X_test, y_test = preprocess_dataset(df, TOTAL_CLIENTS, CLIENT_ID)

    # Build model
    if MODEL.lower() == "lstm":
        model = build_lstm((X_train.shape[1], X_train.shape[2]))
        model_filename = f"{ALGO.capitalize()}_LSTM_{TOTAL_CLIENTS}clients.joblib"
        plot_filename = f"{ALGO.capitalize()}_LSTM_{TOTAL_CLIENTS}clients"
    else:
        model = build_bilstm((X_train.shape[1], X_train.shape[2]))
        model_filename = f"{ALGO.capitalize()}_BiLSTM_{TOTAL_CLIENTS}clients.joblib"
        plot_filename = f"{ALGO.capitalize()}_BiLSTM_{TOTAL_CLIENTS}clients"

    # Initialize client
    if ALGO.lower() == "fedavg":
        client = FedAvgClient(model, X_train, y_train, X_test, y_test, CLIENT_ID)
    else:
        client = FedProxClient(model, X_train, y_train, X_test, y_test, CLIENT_ID)

    # Start Flower client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=client)

    # Save trained model
    joblib.dump(model, model_filename)

    ################ CALCULATING MAE THRESHOLDS AND PLOTTING ###################
    print("\n" + "="*80)
    print("CALCULATING RECONSTRUCTION ERROR THRESHOLDS (MAE-based)")
    print("="*80)

    # 1. Calculate Predictions (Reconstruct sequences)
    print("\nReconstructing Training Data...")
    trainPredict = model.predict(X_train, verbose=1)
    print(f"   Prediction shape: {trainPredict.shape}")
    print(f"   Input shape: {X_train.shape}")

    print("\nReconstructing Test Data...")
    testPredict = model.predict(X_test, verbose=1)

    # 2. Calculate MAE for Training (per sequence)
    print("\nComputing Reconstruction Errors...")
    trainMAE = np.mean(np.abs(trainPredict - X_train), axis=(1, 2))
    print(f"   Train MAE shape: {trainMAE.shape}")
    print(f"   Train MAE stats:")
    print(f"     Mean   : {trainMAE.mean():.6f}")
    print(f"     Median : {np.median(trainMAE):.6f}")
    print(f"     Std    : {trainMAE.std():.6f}")
    print(f"     Min    : {trainMAE.min():.6f}")
    print(f"     Max    : {trainMAE.max():.6f}")

    # 3. Calculate Threshold (Using configured percentile)
    threshold_mae = np.percentile(trainMAE, THRESHOLD_PERCENTILE)
    print(f"\nThreshold ({THRESHOLD_PERCENTILE}th percentile): {threshold_mae:.6f}")

    # 4. Plot Histogram for Train MAE with Threshold
    def plot_train_mae():
        plt.figure(figsize=(12, 6))
        plt.hist(trainMAE, bins=60, alpha=0.7, color='teal', edgecolor='black')
        plt.axvline(threshold_mae, color='r', linestyle='dashed', linewidth=2, 
                   label=f'Threshold ({THRESHOLD_PERCENTILE}%): {threshold_mae:.6f}')
        plt.xlabel('Reconstruction Error (MAE)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Train Reconstruction Error Distribution (Client {CLIENT_ID})', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

    safe_plot(plot_train_mae, f'{plot_filename}_train_mae_histogram.png')

    # 5. Calculate MAE for Testing
    testMAE = np.mean(np.abs(testPredict - X_test), axis=(1, 2))
    print(f"\n   Test MAE stats:")
    print(f"     Mean   : {testMAE.mean():.6f}")
    print(f"     Median : {np.median(testMAE):.6f}")
    print(f"     Std    : {testMAE.std():.6f}")

    # 6. Plot Histogram for Test MAE
    def plot_test_mae():
        plt.figure(figsize=(12, 6))
        plt.hist(testMAE, bins=60, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(threshold_mae, color='r', linestyle='dashed', linewidth=2, 
                   label=f'Train Threshold: {threshold_mae:.6f}')
        plt.xlabel('Reconstruction Error (MAE)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Test Reconstruction Error Distribution (Client {CLIENT_ID})', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

    safe_plot(plot_test_mae, f'{plot_filename}_test_mae_histogram.png')

    # 7. Plot Reconstruction Error Over Time (Training Data)
    def plot_error_over_time():
        plt.figure(figsize=(15, 5))
        plt.plot(trainMAE, linewidth=0.8, alpha=0.7, color='blue', label='Reconstruction Error')
        plt.axhline(threshold_mae, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({THRESHOLD_PERCENTILE}%)')
        plt.title(f"Reconstruction Error Over Time - Training Data (Client {CLIENT_ID})", 
                  fontsize=14, fontweight='bold')
        plt.xlabel("Sequence Index", fontsize=12)
        plt.ylabel("MAE", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

    safe_plot(plot_error_over_time, f'{plot_filename}_error_over_time.png')

    ####################################################################################################

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    # Save threshold to JSON file
    threshold_data = {
        "client_id": CLIENT_ID,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_mae": float(threshold_mae),
        "train_mae_mean": float(trainMAE.mean()),
        "train_mae_median": float(np.median(trainMAE)),
        "train_mae_std": float(trainMAE.std()),
        "train_mae_min": float(trainMAE.min()),
        "train_mae_max": float(trainMAE.max()),
        "test_mae_mean": float(testMAE.mean()),
        "test_mae_median": float(np.median(testMAE)),
        "test_mae_std": float(testMAE.std()),
        "timestamp": datetime.now().isoformat()
    }
    
    with open('client_threshold.json', 'w') as f:
        json.dump(threshold_data, f, indent=2)
    
    print(f"\nThreshold data saved to: client_threshold.json")
    print(f"   Client {CLIENT_ID} Threshold ({THRESHOLD_PERCENTILE}%): {threshold_mae:.6f}")
    print(f"\n   Training anomalies detected: {np.sum(trainMAE > threshold_mae)} / {len(trainMAE)}")
    print(f"   Test anomalies detected: {np.sum(testMAE > threshold_mae)} / {len(testMAE)}")

    print("\nContainer will stay alive indefinitely...")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down...")