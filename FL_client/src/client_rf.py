import os
import argparse
import flwr as fl
import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings

warnings.simplefilter('ignore')

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--ip', help='Provide the IP address', default="0.0.0.0", required=False)
parser.add_argument('--port', help='Provide the Port address', default="8080", required=False)
parser.add_argument('--id', help='Provide the client id', default="1", required=True)
parser.add_argument('--folder', help='Provide the Dataset folder', default='Client_1', type=str, required=False)
args = parser.parse_args()

# Constants
SERVER_ADDR = f"{args.ip}:{args.port}"
FOLDER_LOC = args.folder
CLIENT_ID = args.id

# Load Dataset Function
def load_dataset():
    folder_path = os.path.join('.', 'data', FOLDER_LOC)
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            print("First few rows of the DataFrame:")
            print(df.head())
            print("Column names:")
            print(df.columns)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

# Get Parameters Function
def get_params(model: RandomForestClassifier) -> List[np.ndarray]:
    params = [
        model.n_estimators,
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf,
    ]
    return params

# Set Parameters Function
def set_params(model: RandomForestClassifier, params: List[np.ndarray]) -> RandomForestClassifier:
    model.n_estimators = int(params[0])
    model.max_depth = int(params[1])
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    return model

# Flower Client Class
class FlowerClient(fl.client.NumPyClient):

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def get_parameters(self, config):
        print("Received the parameters.")
        return get_params(self.model)

    def fit(self, parameters, config):
        print("Parameters before setting:", parameters)
        set_params(self.model, parameters)
        print("Parameters after setting:", self.model.get_params())

        self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round {config['server_round']}.")

        trained_params = get_params(self.model)
        print("Trained Parameters:", trained_params)

        return trained_params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        y_pred = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred, labels=[0, 1])

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        line = "-" * 21
        print(line)
        print(f"Accuracy : {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall   : {recall:.8f}")
        print(f"F1 Score : {f1:.8f}")
        print(line)

        return loss, len(self.X_test), {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        }

# Main Function
if __name__ == "__main__":
    # Load Dataset
    X, y = load_dataset()

    # Print Label Distribution
    train_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"Client {CLIENT_ID}:")
    print("Label distribution in the dataset:", train_counts)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and Fit Model
    model = RandomForestClassifier(
        class_weight='balanced',
        criterion='entropy',
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    model.fit(X_train, y_train)

    # Create Flower Client
    flower_client = FlowerClient()
    flower_client.X_train = X_train
    flower_client.y_train = y_train
    flower_client.X_test = X_test
    flower_client.y_test = y_test
    flower_client.model = model

    # Start Client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=flower_client)
