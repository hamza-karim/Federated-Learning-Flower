import os
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import argparse
import time

# Define the default values for command-line arguments
DEFAULT_IP = "0.0.0.0"
DEFAULT_PORT = "8080"
DEFAULT_INPUT_SEQ = 4
DEFAULT_NUM_CLUSTERS = 2
DEFAULT_FOLDER_LOC = 'one-trip'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="A simple command-line")
parser.add_argument('--ip', help='Provide the IP address', default=DEFAULT_IP, required=False)
parser.add_argument('--port', help='Provide the Port address', default=DEFAULT_PORT, required=False)
parser.add_argument('--input_seq', help='Provide the Time-Series Window Size', default=DEFAULT_INPUT_SEQ, type=int, required=False)
parser.add_argument('--num_clusters', help='Provide the Number of Clusters', default=DEFAULT_NUM_CLUSTERS, type=int, required=False)
parser.add_argument('--folder', help='Provide the Dataset folder', default=DEFAULT_FOLDER_LOC, type=str, required=False)
args = parser.parse_args()

# Constants from command-line arguments
SERVER_ADDR = f'{args.ip}:{args.port}'
INPUT_SEQ = args.input_seq
NUM_CLUSTERS = args.num_clusters
FOLDER_LOC = args.folder
COLUMN_NAME = 'Hz'

# Function to read and preprocess the dataset
def read_dataset():
    folder_path = os.path.join('.', 'data', FOLDER_LOC)
    df_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            # Remove datetime column and any other columns not required
            if 'datetime' in df.columns:
                df = df.drop(columns=['datetime'])
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Function to prepare data for training
def prepare_data(df):
    X, y = [], []
    for i in range(len(df) - INPUT_SEQ):
        X.append(df[COLUMN_NAME][i:i+INPUT_SEQ].values)
        y.append(df[COLUMN_NAME][i+INPUT_SEQ])
    return np.array(X), np.array(y)

# Load and preprocess dataset
dataset = read_dataset()
X, y = prepare_data(dataset)

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Flower client class for linear regression model
class LinearRegressionClient(fl.client.NumPyClient):
    def get_parameters(self):
        """Get the current weights of the model."""
        return model.coef_, model.intercept_

    def fit(self, parameters, config):
        """Set the given weights to the model."""
        weights, bias = parameters
        model.coef_ = weights
        model.intercept_ = bias
        return self.get_parameters()

    def evaluate(self, parameters, config):
        """Evaluate the model on the test set."""
        weights, bias = parameters
        model.coef_ = weights
        model.intercept_ = bias
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        return float(mae), len(X_test)

# Start Flower client
fl.client.start_numpy_client(server_address=SERVER_ADDR, client=LinearRegressionClient())
