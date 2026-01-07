# Muhammad Hamza Karim

import os
# Force CPU/GPU threading limits to prevent resource fighting
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "default"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import joblib
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import random
import time

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import faulthandler
import warnings
from keras import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Dict

warnings.simplefilter('ignore')
faulthandler.enable()

# ---------------- Environment Variables ---------------- #
# The dashboard sends these via -e flags
SERVER_IP = os.getenv("SERVER_IP", "10.226.47.97")
SERVER_PORT = os.getenv("SERVER_PORT", "8080")
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))  # <--- CRITICAL: Reads ID from Dashboard
TOTAL_CLIENTS = int(os.getenv("TOTAL_CLIENTS", "20"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
MODEL = os.getenv("MODEL", "lstm")
ALGO = os.getenv("ALGO", "fedavg")

SERVER_ADDR = f"{SERVER_IP}:{SERVER_PORT}"
temp_loss = []
temp_mape = []

# Set random seed
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ---------------- Dataset Loading ---------------- #
def load_dataset():
    folder_path = './Train_data'  
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)
            dataframe['datetimeCST'] = pd.to_datetime(dataframe['datetimeCST'])
            df = dataframe[['datetimeCST', 'Hz_mod']]
            df.set_index('datetimeCST', inplace=True)

    # FIX 1: Convert to NumPy immediately to avoid SegFault in Matplotlib
    x_vals = df.index.to_numpy()

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, df['Hz_mod'].values, linewidth=1)
        plt.title("Frequency Variation (Hz) Over One Week", fontsize=14, fontweight='bold')
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.gcf().autofmt_xdate()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("frequency_week_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Warning: plotting failed in load_dataset():", e)

    return df

# ---------------- Dataset Preprocessing ---------------- #
def preprocess_dataset(df: pd.DataFrame, total_clients: int, client_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal_data = df.copy()
    total_rows = len(normal_data)
    rows_per_client = total_rows // total_clients

    # Plot slices for all clients
    try:
        plt.figure(figsize=(12, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, total_clients))
        for i in range(total_clients):
            start_idx = i * rows_per_client
            end_idx = start_idx + rows_per_client if i != total_clients - 1 else total_rows
            client_slice = normal_data.iloc[start_idx:end_idx]
            
            # FIX 1: Safe plotting
            x_vals = client_slice.index.to_numpy()
            plt.plot(x_vals, client_slice['Hz_mod'].values, color=colors[i], label=f'Client {i+1}', linewidth=1)
            
        plt.title(f"Frequency Variation (Hz) Divided for {total_clients} Clients", fontsize=14, fontweight='bold')
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"frequency_divided_{total_clients}_clients.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Global plotting failed: {e}")

    # Slice for this client
    start_idx = (client_id - 1) * rows_per_client
    end_idx = start_idx + rows_per_client if client_id != total_clients else total_rows
    client_data = normal_data.iloc[start_idx:end_idx]

    # ---------------- Plot Client Dataset ---------------- #
    try:
        plt.figure(figsize=(12, 6))
        x_vals = client_data.index.to_numpy()
        plt.plot(x_vals, client_data['Hz_mod'].values, color='teal', linewidth=1)
        plt.title(f"Frequency Variation (Hz) for Client {client_id}", fontsize=14, fontweight='bold')
        plt.xlabel("Time (Days)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"client_{client_id}_frequency_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Client plotting failed: {e}")

    # Train/test split
    train_size = int(0.9 * len(client_data))
    train = client_data.iloc[:train_size]
    test = client_data.iloc[train_size:]

    seq_size = 20
    def to_sequence(x, y, seq_size=1):
        x_vals, y_vals = [], []
        for i in range(len(x) - seq_size):
            x_vals.append(x.iloc[i:(i + seq_size)].values)
            y_vals.append(y.iloc[i + seq_size])
        return np.array(x_vals), np.array(y_vals)

    X_train, y_train = to_sequence(train[['Hz_mod']], train['Hz_mod'], seq_size)
    X_test, y_test = to_sequence(test[['Hz_mod']], test['Hz_mod'], seq_size)

    return X_train, y_train, X_test, y_test

# ---------------- Model Builders ---------------- #
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer='adam', loss='mae', metrics=["mape"])
    return model

def build_bilstm(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)))
    model.add(RepeatVector(input_shape[0]))
    model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
    model.add(Bidirectional(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer='adam', loss='mae', metrics=["mape"])
    return model

# ---------------- Flower Clients (Updated with get_properties) ---------------- #
class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id  # Store ID

    # FIX 2: Implement get_properties so server knows who this is
    def get_properties(self, config):
        return {"client_id": int(self.client_id)}

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2, verbose=1)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mape = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        temp_loss.append(loss)
        temp_mape.append(mape)
        print(f"Eval Loss: {loss} || Eval MAPE: {mape}")
        return loss, len(self.X_test), {"mape": mape}


class FedProxClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id # Store ID
        self.global_weights = None
        self.mu = 0.0

    # FIX 2: Implement get_properties
    def get_properties(self, config):
        return {"client_id": int(self.client_id)}

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.global_weights = parameters
        self.mu = config.get("proximal_mu", 0.0)
        self.model.compile(optimizer="adam", loss=self.fedprox_loss)
        self.model.fit(self.X_train, self.y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2, verbose=1)
        return self.model.get_weights(), len(self.X_train), {}

    def fedprox_loss(self, y_true, y_pred):
        base_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        prox_term = 0.0
        for w, w0 in zip(self.model.trainable_weights, self.global_weights):
            prox_term += tf.reduce_sum(tf.square(w - tf.convert_to_tensor(w0)))
        return base_loss + (self.mu / 2.0) * prox_term

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(optimizer="adam", loss="mae", metrics=["mape"])
        loss, mape = self.model.evaluate(self.X_test, self.y_test)
        temp_loss.append(loss)
        temp_mape.append(mape)
        print(f"Eval Loss: {loss} || Eval MAPE: {mape}")
        return loss, len(self.X_test), {"mape": mape}


# ---------------- Main ---------------- #
if __name__ == "__main__":
    print(f"\n===== Client {CLIENT_ID} Configuration =====")
    print(f"Server Address : {SERVER_ADDR}")
    print(f"Total Clients  : {TOTAL_CLIENTS}")
    print(f"Model Selected : {MODEL}")
    print(f"Iteration Per Round : {EPOCHS}")
    print(f"Algorithm      : {ALGO.capitalize()}")
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

    # Initialize client, passing CLIENT_ID
    if ALGO.lower() == "fedavg":
        client = FedAvgClient(model, X_train, y_train, X_test, y_test, CLIENT_ID)
    else:
        client = FedProxClient(model, X_train, y_train, X_test, y_test, CLIENT_ID)

    # Start Flower client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=client)

    # Save trained model
    joblib.dump(model, model_filename)

    ################ CALCULATING MAPE THRESHOLDS AND PLOTTING ###################

    # 1. Calculate Predictions
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    # 2. Calculate MAPE for Training
    trainActual = X_train
    trainMAPE = np.mean(np.abs(trainPredict - trainActual) / trainActual, axis=1) * 100
    print("Mean of Train MAPE:", np.mean(trainMAPE))

    # --- CALCULATE THRESHOLD (99th Percentile of Training MAPE) ---
    threshold_mape = np.percentile(trainMAPE, 99)
    print(f"Calculated 99th Percentile Threshold (MAPE): {threshold_mape}")

    # 3. Plot Histogram for Train MAPE with Threshold
    plt.figure(figsize=(10, 6))
    plt.hist(trainMAPE, bins=30, alpha=0.7, color='teal', edgecolor='black')
    plt.axvline(threshold_mape, color='r', linestyle='dashed', linewidth=2, label=f'Threshold (99th): {threshold_mape:.2f}%')
    plt.xlabel('Mean Absolute Percentage Error (MAPE)')
    plt.ylabel('Frequency')
    plt.title(f'Train MAPE Histogram (Client {CLIENT_ID})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plot_filename}_train_mape_histogram.png')
    plt.close()

    # 4. Calculate MAPE for Testing
    testActual = X_test
    testMAPE = np.mean(np.abs(testPredict - testActual) / testActual, axis=1) * 100
    print("Mean of Test MAPE:", np.mean(testMAPE))

    # 5. Plot Histogram for Test MAPE (Using the Training Threshold)
    plt.figure(figsize=(10, 6))
    plt.hist(testMAPE, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(threshold_mape, color='r', linestyle='dashed', linewidth=2, label=f'Train Threshold: {threshold_mape:.2f}%')
    plt.xlabel('Mean Absolute Percentage Error (MAPE)')
    plt.ylabel('Frequency')
    plt.title(f'Test MAPE Histogram (Client {CLIENT_ID})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plot_filename}_test_mape_histogram.png')
    plt.close()

    ####################################################################################################

    print("\n" + "="*60)
    print("Training completed! Container will stay alive indefinitely.")
    print("="*60 + "\n")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down...")