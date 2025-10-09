# Muhammad Hamza Karim

import os
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
from typing import Tuple
warnings.simplefilter('ignore')
faulthandler.enable()

# # ---------------- Argument Parser ---------------- #
# parser = argparse.ArgumentParser()
# parser.add_argument('--ip', help='Provide the IP address', default="10.226.47.85", required=False)
# parser.add_argument('--port', help='Provide the Port address', default="8080", required=False)
# args = parser.parse_args()

# # ---------------- Interactive Inputs ---------------- #
# CLIENT_ID = int(input("Enter the client id: "))
# TOTAL_CLIENTS = int(input("Enter total number of clients: "))

# # Number of epochs
# try:
#     EPOCHS = int(input("Enter number of epochs for local training (e.g., 5): "))
# except ValueError:
#     print("Invalid input. Using default: 5 epochs.")
#     EPOCHS = 5

# # Model choice
# while True:
#     model_choice = input("Select model: 1 for LSTM, 2 for BiLSTM: ").lower()
#     if model_choice in ['1', '2', 'lstm', 'bilstm']:
#         if model_choice == '1':
#             model_choice = 'lstm'
#         elif model_choice == '2':
#             model_choice = 'bilstm'
#         break
#     print("Invalid choice. Please enter 1, 2, 'lstm', or 'bilstm'.")

# # Aggregation algorithm
# while True:
#     algo_choice = input("Select FL aggregation algorithm: 1 for FedAvg, 2 for FedProx: ")
#     if algo_choice in ['1', '2']:
#         break
#     print("Invalid choice. Enter 1 or 2.")

SERVER_IP = os.getenv("SERVER_IP", "10.226.47.97")
SERVER_PORT = os.getenv("SERVER_PORT", "8080")
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
TOTAL_CLIENTS = int(os.getenv("TOTAL_CLIENTS", "20"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
MODEL = os.getenv("MODEL", "lstm")
ALGO = os.getenv("ALGO", "fedavg")

# ---------------- Constants ---------------- #
SERVER_ADDR = f"{SERVER_IP}:{SERVER_PORT}"
# SERVER_ADDR = f"{args.ip}:{args.port}"
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

    # Convert index to python datetime objects for safe plotting
    x_vals = df.index.to_pydatetime()

    # Plot dataset (guarded)
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, df['Hz_mod'], linewidth=1)
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
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, total_clients))
    for i in range(total_clients):
        start_idx = i * rows_per_client
        end_idx = start_idx + rows_per_client if i != total_clients - 1 else total_rows
        client_slice = normal_data.iloc[start_idx:end_idx]
        plt.plot(client_slice.index, client_slice['Hz_mod'], color=colors[i], label=f'Client {i+1}', linewidth=1)
    plt.title(f"Frequency Variation (Hz) Divided for {total_clients} Clients", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"frequency_divided_{total_clients}_clients.png", dpi=300, bbox_inches='tight')
   # plt.show()

    # Slice for this client
    start_idx = (client_id - 1) * rows_per_client
    end_idx = start_idx + rows_per_client if client_id != total_clients else total_rows
    client_data = normal_data.iloc[start_idx:end_idx]

    # ---------------- Print Client Info ---------------- #
    print(f"\n===== Client {client_id} Dataset Info =====")
    print(f"Shape: {client_data.shape}")
    print("First 5 rows:")
    print(client_data.head())
    print("Last 5 rows:")
    print(client_data.tail())
    print("\nSummary Statistics:")
    print(client_data.describe())

    # ---------------- Plot Client Dataset ---------------- #
    plt.figure(figsize=(12, 6))
    plt.plot(client_data.index, client_data['Hz_mod'], color='teal', linewidth=1)
    plt.title(f"Frequency Variation (Hz) for Client {client_id}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"client_{client_id}_frequency_plot.png", dpi=300, bbox_inches='tight')
    #plt.show()

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

# ---------------- Flower Clients ---------------- #
class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

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
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.global_weights = None
        self.mu = 0.0

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

    # Initialize client
    if ALGO.lower() == "fedavg":
        client = FedAvgClient(model, X_train, y_train, X_test, y_test)
    else:
        client = FedProxClient(model, X_train, y_train, X_test, y_test)

    # Start Flower client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=client)

    # Save trained model
    joblib.dump(model, model_filename)


################ CALCULATING THE MAE AND MAPE FOR TRAIN AND TEST FOR THRESHOLDING ###################

    # Calculate MAE for training prediction
    trainPredict = model.predict(X_train)
    trainMAE = np.mean(np.abs(trainPredict - X_train), axis=1)
    print("Mean of Train MAE:", np.mean(trainMAE))

    # Plot
    # plt.figure(figsize=(8, 6))
    # plt.hist(trainMAE, bins=30)
    # plt.xlabel('Mean Absolute Error (MAE)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mean Absolute Error (MAE) in Training Prediction')
    # plt.savefig(f'{plot_filename}_train_mae_histogram.png')
    # plt.close()

    # Calculate MAPE for each sample
    trainActual = X_train
    trainMAPE = np.mean(np.abs(trainPredict - trainActual) / trainActual, axis=1) * 100

    # Print the mean of MAPE
    print("Mean of Train MAPE:", np.mean(trainMAPE))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(trainMAPE, bins=30)
    plt.xlabel('Mean Absolute Percentage Error (MAPE)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mean Absolute Percentage Error (MAPE) in Training Prediction')
    plt.savefig(f'{plot_filename}_train_mape_histogram.png')
    plt.close()

    # Calculate reconstruction loss (MAE) for testing dataset
    testPredict = model.predict(X_test)
    testMAE = np.mean(np.abs(testPredict - X_test), axis=1)

    # Print the mean of test MAE
    print("Mean of Test MAE:", np.mean(testMAE))

    # Plot histogram
    # plt.figure(figsize=(8, 6))
    # plt.hist(testMAE, bins=30)
    # plt.xlabel('Test MAE')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mean Absolute Error (MAE) in Test Prediction')
    # plt.savefig(f'{plot_filename}_test_mae_histogram.png')
    # plt.close()

    # Calculate MAPE for each sample
    testActual = X_test
    testMAPE = np.mean(np.abs(testPredict - testActual) / testActual, axis=1) * 100

    # Print the mean of MAPE
    print("Mean of Test MAPE:", np.mean(testMAPE))

    # Plot histogram of MAPE
    plt.figure(figsize=(8, 6))
    plt.hist(testMAPE, bins=30)
    plt.xlabel('Mean Absolute Percentage Error (MAPE)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mean Absolute Percentage Error (MAPE) in Test Prediction')
    plt.savefig(f'{plot_filename}_test_mape_histogram.png')
    plt.close()

####################################################################################################