# Muhammad Hamza Karim - FedAvg Version
import os
import joblib
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import keras as ke
from typing import Tuple
import warnings
from keras import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

# ---------------- Argument Parser ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--ip', help='Provide the IP address', default="10.226.47.254", required=False)
parser.add_argument('--port', help='Provide the Port address', default="8080", required=False)
args = parser.parse_args()

# ---------------- Interactive Inputs ---------------- #
CLIENT_ID = int(input("Enter the client id: "))
TOTAL_CLIENTS = int(input("Enter total number of clients: "))
while True:
    model_choice = input("Specify which model to train (lstm or bilstm): ").lower()
    if model_choice in ['lstm', 'bilstm']:
        break
    else:
        print("Invalid choice. Please enter 'lstm' or 'bilstm'.")

# ---------------- Constants ---------------- #
SERVER_ADDR = f"{args.ip}:{args.port}"

# Print chosen configuration
print(f"\n===== Client {CLIENT_ID} Configuration =====")
print(f" Server Address : {SERVER_ADDR}")
print(f" Client ID      : {CLIENT_ID}")
print(f" Total Clients  : {TOTAL_CLIENTS}")
print(f" Model Selected : {model_choice}")
print("=========================\n")

temp_loss = []
temp_mape = []

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load Dataset Function
def load_dataset():
    folder_path = os.path.join('.', 'Train_data')
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)
            dataframe['datetimeCST'] = pd.to_datetime(dataframe['datetimeCST'])  # Convert to datetime
            df = dataframe[['datetimeCST', 'Hz_mod']]  # Select only required columns
            df.set_index('datetimeCST', inplace=True)  # Set 'datetimestamp' as index

    # Print preview
    print("First few rows of the DataFrame:")
    print(df.head())
    print("Column names:")
    print(df.columns)

    # -------- Plotting -------- #
    plt.figure(figsize=(12, 6)) 
    plt.plot(df.index, df['Hz_mod'], color='navy', linewidth=1)
    plt.title("Frequency Variation (Hz) Over One Week", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)

    # X-axis scaling 
    plt.gcf().autofmt_xdate()  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig("frequency_week_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    return df

def preprocess_dataset(df: pd.DataFrame, total_clients: int, client_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Start date of dataset:", df.index.min())
    print("End date of dataset:", df.index.max())

    normal_data = df.copy()
    total_rows = len(normal_data)
    rows_per_client = total_rows // total_clients

    # Plot dataset slices for each client
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
    plt.show()

    # Determine slice for this client (time-contiguous)
    start_idx = (client_id - 1) * rows_per_client
    end_idx = start_idx + rows_per_client if client_id != total_clients else total_rows
    client_data = normal_data.iloc[start_idx:end_idx]

    # Split into train (90%) and test (10%)
    train_size = int(0.9 * len(client_data))
    train = client_data.iloc[:train_size]
    test = client_data.iloc[train_size:]

    print(f"Client {client_id} | Train rows: {len(train)} | Test rows: {len(test)}")

    seq_size = 20  # Number of time steps
    def to_sequence(x, y, seq_size=1):
        x_values, y_values = [], []
        for i in range(len(x) - seq_size):
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i + seq_size])
        return np.array(x_values), np.array(y_values)

    X_train, y_train = to_sequence(train[['Hz_mod']], train['Hz_mod'], seq_size)
    X_test, y_test = to_sequence(test[['Hz_mod']], test['Hz_mod'], seq_size)

    print("X_train shape:", X_train.shape, "| y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape, "| y_test shape:", y_test.shape)

    return X_train, y_train, X_test, y_test


def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                   input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid',
                   return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid',
                   return_sequences=True))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid',
                   return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer='adam', loss='mae', metrics=["mape"])
    return model


def build_bilstm(input_shape):
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
    model.compile(optimizer='adam', loss='mae', metrics=["mape"])
    return model


class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r= self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=100, validation_split=0.2, verbose=1)
        hist = r.history
        print("Fit history : ", hist)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        eval_loss, eval_mape = self.model.evaluate(X_test, y_test)

        temp_loss.append(eval_loss)
        temp_mape.append(eval_mape)

        print(f"Eval Loss: {eval_loss} || Eval MAPE: {eval_mape}")
        return eval_loss, len(X_test), {"mape": eval_mape}


# Main Function
if __name__ == "__main__":
    # Load Dataset
    df = load_dataset()
    # Preprocess/split Dataset
    X_train, y_train, X_test, y_test = preprocess_dataset(df, TOTAL_CLIENTS, CLIENT_ID)

    # Select model
    if model_choice == "lstm":
        model = build_lstm((X_train.shape[1], X_train.shape[2]))
        model_filename = f"FedAvg_LSTM_{TOTAL_CLIENTS}clients.joblib"
        plot_filename = f"FedAvg_LSTM_{TOTAL_CLIENTS}clients"
    elif model_choice == "bilstm":
        model = build_bilstm((X_train.shape[1], X_train.shape[2]))
        model_filename = f"FedAvg_BiLSTM_{TOTAL_CLIENTS}clients.joblib"
        plot_filename = f"FedAvg_BiLSTM_{TOTAL_CLIENTS}clients"


    # Create Flower Client
    flower_client = FlowerClient()
    flower_client.X_train = X_train
    flower_client.y_train = y_train
    flower_client.X_test = X_test
    flower_client.y_test = y_test
    flower_client.model = model

    # Start Client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=flower_client)

    # Save the trained model
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