import argparse
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time
import os
import joblib

print("--------------------")
start_time = time.time()
print(f"START TIME - {start_time}")

temp_accuracy = []
temp_precision = []
temp_recall = []
temp_f1 = []

parser = argparse.ArgumentParser(description="A simple command-line")

# Add arguments
parser.add_argument('--ip',
                    help='Provide the IP address',
                    default="0.0.0.0",
                    required=False)
parser.add_argument('--port',
                    help='Provide the Port address',
                    default="8080",
                    required=False)
parser.add_argument('--input_seq',
                    help='Provide the Time-Series Window Size',
                    default=4,
                    type=str,
                    required=False)
parser.add_argument('--num_clusters',
                    help='Provide the Number of Clusters',
                    default=2,
                    type=int,
                    required=False)
parser.add_argument('--folder',
                    help='Provide the Dataset folder',
                    default='one-trip',
                    type=str,
                    required=False)

args = parser.parse_args()

SERVER_ADDR = f'{args.ip}:{args.port}'
INPUT_SEQ = args.input_seq
NUM_CLUSTERS = args.num_clusters
FOLDER_LOC = args.folder


def read_uni_dataset(dataf):
    df_np = dataf.to_numpy()
    df_X = []
    df_y = []

    for i in range(len(df_np) - INPUT_SEQ):
        row = [a for a in df_np[i:i + INPUT_SEQ]]
        df_X.append(row)
        label = df_np[i + INPUT_SEQ]
        df_y.append(label)
    return np.array(df_X), np.array(df_y)


def convert_to_train_test():
    folder_path = os.path.join('.', 'data', f'{FOLDER_LOC}')
    temp_store = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            print(df.head())  # Print the first few rows of the DataFrame
            print(df.columns)  # Print column names
            temp_store = df

    # Drop the "datetimestamp" column
    temp_store.drop(columns=["datetimeCST"], inplace=True)

    # Define features (X) and target variable (y)
    X = temp_store.drop(columns=["mod_BIN", "Hz_mod", "PF"])
    y = temp_store["mod_BIN"]

    # Shuffle the data
    temp_store = temp_store.sample(frac=1, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = convert_to_train_test()

    ################ RandomForest #################

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)


    ################ RandomForest #################

    class RandomForestClient(fl.client.NumPyClient):
        def __init__(self, model) -> None:
            self.model = model

        def get_parameters(self, config):
            return []  # No need to return any parameters for this model

        def fit(self, parameters, config):
            self.model.fit(X_train, y_train)  # Fit the model using training data
            return self.get_parameters(config), len(X_train), {}

        def evaluate(self, parameters, config):
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print("accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            temp_accuracy.append(accuracy)
            temp_precision.append(precision)
            temp_recall.append(recall)
            temp_f1.append(f1)
            return float(0), len(X_test), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


    fl.client.start_numpy_client(server_address=SERVER_ADDR,
                                 client=RandomForestClient(model))

    # Save the trained model to a file
    joblib.dump(model, 'trained_model.joblib')

    print(f'Accuracy - {temp_accuracy}')
    print(f'Precision - {temp_precision}')
    print(f'Recall - {temp_recall}')
    print(f'F1 Score - {temp_f1}')

    end_time = time.time()
    print("--------------------")
    print(f"END TIME - {end_time}")

    print(f"ELAPSED TIME -  {end_time - start_time}")