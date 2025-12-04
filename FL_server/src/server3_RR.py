# server3_RR_fixed.py
# Muhammad Hamza Karim
# Enhanced Server with Communication and Compute Metrics (fixed)

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "default"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import argparse
import flwr as fl
import numpy as np
import faulthandler
from typing import Dict, List, Tuple, Optional
import time
import sys
import json
from collections import defaultdict
faulthandler.enable()

# -------------------- Metrics Tracker --------------------
class MetricsTracker:
    def __init__(self):
        self.round_metrics = defaultdict(dict)
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.round_start_times = {}
        self.round_end_times = {}
        self.communication_rounds = 0

    def start_round(self, round_num: int):
        self.round_start_times[round_num] = time.time()

    def end_round(self, round_num: int):
        self.round_end_times[round_num] = time.time()

    def get_round_duration(self, round_num: int) -> float:
        if round_num in self.round_start_times and round_num in self.round_end_times:
            return self.round_end_times[round_num] - self.round_start_times[round_num]
        return 0.0

    def calculate_parameters_size(self, parameters) -> int:
        """Calculate size of parameters in bytes.

        Supports:
        - Flower Parameters object (has .tensors where each tensor is bytes)
        - list of numpy arrays
        - numpy array
        - fallback: sys.getsizeof
        """
        total_size = 0
        if parameters is None:
            return 0

        # Flower Parameters proto-like object with .tensors (list of bytes)
        if hasattr(parameters, "tensors"):
            try:
                for tensor in parameters.tensors:
                    # If each tensor is raw bytes-like
                    if isinstance(tensor, (bytes, bytearray)):
                        total_size += len(tensor)
                    else:
                        # fallback: getsizeof
                        total_size += sys.getsizeof(tensor)
            except Exception:
                total_size = sys.getsizeof(parameters)
        elif isinstance(parameters, list):
            for param in parameters:
                if isinstance(param, np.ndarray):
                    total_size += int(param.nbytes)
                else:
                    total_size += sys.getsizeof(param)
        elif isinstance(parameters, np.ndarray):
            total_size = int(parameters.nbytes)
        else:
            total_size = sys.getsizeof(parameters)

        return int(total_size)

    def set_bytes_sent_outgoing(self, round_num: int, bytes_sent: int):
        """Store the bytes that were sent to clients at the start of the round."""
        self.round_metrics[round_num]['bytes_sent_outgoing'] = int(bytes_sent)

    def add_communication_overhead(self, round_num: int, bytes_sent: int, bytes_received: int, num_clients: int):
        """Record the finalized communication numbers for the round."""
        # Use the already-stored outgoing bytes if present (preferable)
        if 'bytes_sent_outgoing' in self.round_metrics[round_num]:
            bytes_sent = int(self.round_metrics[round_num]['bytes_sent_outgoing'])
        else:
            bytes_sent = int(bytes_sent)

        self.round_metrics[round_num]['bytes_sent'] = int(bytes_sent)
        self.round_metrics[round_num]['bytes_received'] = int(bytes_received)
        self.round_metrics[round_num]['total_bytes'] = int(bytes_sent) + int(bytes_received)
        self.round_metrics[round_num]['num_clients_communicated'] = int(num_clients)

        self.total_bytes_sent += int(bytes_sent)
        self.total_bytes_received += int(bytes_received)
        self.communication_rounds += 1

    def add_compute_metrics(self, round_num: int, aggregation_time: float, num_parameters: int, num_parameter_bytes: int):
        self.round_metrics[round_num]['aggregation_time'] = float(aggregation_time)
        self.round_metrics[round_num]['num_parameters'] = int(num_parameters)  # number of elements
        self.round_metrics[round_num]['num_parameter_bytes'] = int(num_parameter_bytes)  # size in bytes

    def print_round_summary(self, round_num: int):
        metrics = self.round_metrics.get(round_num, {})
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} SUMMARY")
        print(f"{'='*60}")
        print(f"Round Duration: {self.get_round_duration(round_num):.2f} seconds")

        if 'bytes_sent' in metrics or 'bytes_received' in metrics:
            bytes_sent = metrics.get('bytes_sent', metrics.get('bytes_sent_outgoing', 0))
            bytes_received = metrics.get('bytes_received', 0)
            total_bytes = metrics.get('total_bytes', bytes_sent + bytes_received)
            num_clients = metrics.get('num_clients_communicated', 0)

            print(f"\nCommunication Metrics:")
            print(f"  - Bytes Sent to Clients: {bytes_sent:,} bytes ({bytes_sent/1024/1024:.2f} MB)")
            print(f"  - Bytes Received from Clients: {bytes_received:,} bytes ({bytes_received/1024/1024:.2f} MB)")
            print(f"  - Total Communication: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
            print(f"  - Number of Clients Communicated: {num_clients}")
            if num_clients > 0:
                print(f"  - Avg Communication per Client: {total_bytes/num_clients:,.0f} bytes")

        if 'aggregation_time' in metrics:
            print(f"\nCompute Metrics:")
            print(f"  - Aggregation Time: {metrics['aggregation_time']:.4f} seconds")
            print(f"  - Number of Parameters (elements): {metrics.get('num_parameters', 0):,}")
            print(f"  - Number of Parameters (bytes): {metrics.get('num_parameter_bytes', 0):,} bytes")
        print(f"{'='*60}\n")

    def print_final_summary(self):
        print(f"\n{'='*60}")
        print(f"FEDERATED LEARNING FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total Communication Rounds: {self.communication_rounds}")
        print(f"Total Bytes Sent: {self.total_bytes_sent:,} bytes ({self.total_bytes_sent/1024/1024:.2f} MB)")
        print(f"Total Bytes Received: {self.total_bytes_received:,} bytes ({self.total_bytes_received/1024/1024:.2f} MB)")
        print(f"Total Communication Overhead: {(self.total_bytes_sent + self.total_bytes_received):,} bytes ({(self.total_bytes_sent + self.total_bytes_received)/1024/1024:.2f} MB)")

        total_time = sum([self.get_round_duration(r) for r in self.round_start_times.keys()])
        print(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

        if self.communication_rounds > 0:
            avg_bytes_per_round = (self.total_bytes_sent + self.total_bytes_received) / self.communication_rounds
            print(f"Average Communication per Round: {avg_bytes_per_round:,.0f} bytes ({avg_bytes_per_round/1024/1024:.2f} MB)")

        print(f"\nPer-Round Breakdown:")
        for round_num in sorted(self.round_metrics.keys()):
            metrics = self.round_metrics[round_num]
            duration = self.get_round_duration(round_num)
            comm = metrics.get('total_bytes', metrics.get('bytes_sent_outgoing', 0) + metrics.get('bytes_received', 0))
            print(f"  Round {round_num}: {duration:.2f}s | {comm:,} bytes ({comm/1024/1024:.2f} MB)")

        print(f"{'='*60}\n")

        # Save to JSON file
        self.save_metrics_to_file()

    def save_metrics_to_file(self):
        """Save all metrics to a JSON file"""
        output = {
            'summary': {
                'total_communication_rounds': self.communication_rounds,
                'total_bytes_sent': self.total_bytes_sent,
                'total_bytes_received': self.total_bytes_received,
                'total_communication_overhead': self.total_bytes_sent + self.total_bytes_received,
                'total_training_time': sum([self.get_round_duration(r) for r in self.round_start_times.keys()])
            },
            'per_round_metrics': {}
        }

        for round_num in sorted(self.round_metrics.keys()):
            output['per_round_metrics'][f'round_{round_num}'] = {
                'duration_seconds': self.get_round_duration(round_num),
                **self.round_metrics[round_num]
            }

        with open('fl_metrics.json', 'w') as f:
            json.dump(output, f, indent=4)

        print("Metrics saved to fl_metrics.json")


# Global metrics tracker
metrics_tracker = MetricsTracker()

# -------------------- Argument Parsing --------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Embedded devices - Choose FL strategy")
    parser.add_argument("--ip", help="Provide the IP address", default="0.0.0.0", required=False)
    parser.add_argument("--port", help="Provide the Port address", default="8080", required=False)
    args = parser.parse_args()

    # Strategy selection
    strategy_choice = input("Choose FL strategy (1=FedAvg, 2=FedProx): ").strip()
    args.strategy = "fedavg" if strategy_choice == "1" else "fedprox"

    # Number of rounds
    try:
        args.num_rounds = int(input("Enter the number of federated learning rounds (e.g., 5): "))
    except Exception:
        print("Invalid input. Using default: 5 rounds.")
        args.num_rounds = 5

    # Total number of clients configuration
    print("\nClient Configuration Options:")
    print("1. 4 clients")
    print("2. 8 clients")
    print("3. 12 clients")
    print("4. 16 clients")
    print("5. 20 clients")
    print("6. Custom")

    client_choice = input("Choose total number of clients (1-6): ").strip()
    client_options = {"1": 4, "2": 8, "3": 12, "4": 16, "5": 20}

    if client_choice in client_options:
        args.total_clients = client_options[client_choice]
    elif client_choice == "6":
        try:
            args.total_clients = int(input("Enter custom number of clients: "))
        except Exception:
            print("Invalid input. Using default: 4 clients.")
            args.total_clients = 4
    else:
        print("Invalid choice. Using default: 4 clients.")
        args.total_clients = 4

    # Sample fraction
    try:
        args.sample_fraction = float(input(f"Enter fraction of clients to sample from {args.total_clients} clients (0.0-1.0): "))
        if args.sample_fraction < 0.0 or args.sample_fraction > 1.0:
            raise ValueError()
    except Exception:
        print("Invalid input. Using default: 0.5")
        args.sample_fraction = 0.5

    # Calculate actual number of clients to be selected
    args.selected_clients = max(1, int(args.total_clients * args.sample_fraction))

    # Minimum clients for fit/evaluate (should be <= selected_clients)
    try:
        default_min = min(args.selected_clients, args.total_clients)
        args.min_fit_clients = int(input(f"Enter minimum clients for training (max {default_min}): ") or str(default_min))
        args.min_evaluate_clients = int(input(f"Enter minimum clients for evaluation (max {default_min}): ") or str(default_min))

        # Ensure minimums don't exceed selected clients
        args.min_fit_clients = min(args.min_fit_clients, args.selected_clients)
        args.min_evaluate_clients = min(args.min_evaluate_clients, args.selected_clients)
    except Exception:
        args.min_fit_clients = args.selected_clients
        args.min_evaluate_clients = args.selected_clients

    # FedProx-specific parameter
    if args.strategy == "fedprox":
        try:
            args.proximal_mu = float(input("Enter FedProx proximal term mu (e.g., 0.1): "))
        except Exception:
            print("Invalid input. Using default: 0.1")
            args.proximal_mu = 0.1

    return args


# -------------------- Metrics Aggregation --------------------
def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    mape = [num_examples * m["mape"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"mape": sum(mape) / sum(examples)}


# -------------------- Custom Evaluation Function --------------------
def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters, config):
        print(f"[Server] Server-side evaluation round {server_round}")
        return None
    return evaluate


# -------------------- Strategies with Metrics Tracking --------------------
class BaseRoundRobinStrategy:
    def wait_for_clients(self, client_manager, min_clients):
        available_clients = client_manager.all()
        while len(available_clients) < min_clients:
            print(f"[Server] Waiting for clients to connect... {len(available_clients)}/{min_clients}")
            time.sleep(2)
            available_clients = client_manager.all()
        return available_clients


class SaveModelFedAvgStrategy(BaseRoundRobinStrategy, fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_fit_index = 0
        self._last_eval_index = 0

    def configure_fit(self, server_round, parameters, client_manager):
        # Start tracking round (timing)
        metrics_tracker.start_round(server_round)

        config = {"server_round": server_round}
        available_clients = self.wait_for_clients(client_manager, self.min_available_clients)
        client_ids = list(available_clients.keys())
        num_connected = len(client_ids)

        sample_size, _ = self.num_fit_clients(num_connected)

        # Round-robin selection
        selected_clients = []
        for i in range(sample_size):
            index = (self._last_fit_index + i) % num_connected
            selected_clients.append(available_clients[client_ids[index]])
        self._last_fit_index = (self._last_fit_index + sample_size) % num_connected
        selected_ids = [c.cid for c in selected_clients]

        print(f"\n[Server] Round {server_round} - Connected clients: {num_connected}")
        print(f"[Server] Client IDs: {client_ids}")
        print(f"[Server] Sample fraction: {self.fraction_fit} → Selecting {sample_size} clients for training")
        print(f"[Server] Selected clients for training (round-robin): {selected_ids}")

        # Calculate bytes sent (parameters to clients) and store it for later use
        bytes_sent = metrics_tracker.calculate_parameters_size(parameters) * sample_size
        print(f"[Server] Sending {bytes_sent:,} bytes ({bytes_sent/1024/1024:.2f} MB) to {sample_size} clients")
        metrics_tracker.set_bytes_sent_outgoing(server_round, bytes_sent)

        return [(client, fl.common.FitIns(parameters, config)) for client in selected_clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {"server_round": server_round}
        available_clients = self.wait_for_clients(client_manager, self.min_available_clients)
        client_ids = list(available_clients.keys())
        num_connected = len(client_ids)

        sample_size, _ = self.num_evaluation_clients(num_connected)

        selected_clients = []
        for i in range(sample_size):
            index = (self._last_eval_index + i) % num_connected
            selected_clients.append(available_clients[client_ids[index]])
        self._last_eval_index = (self._last_eval_index + sample_size) % num_connected
        selected_ids = [c.cid for c in selected_clients]

        print(f"[Server] Sample fraction: {self.fraction_evaluate} → Selecting {sample_size} clients for evaluation")
        print(f"[Server] Selected clients for evaluation (round-robin): {selected_ids}")

        return [(client, fl.common.EvaluateIns(parameters, config)) for client in selected_clients]

    def aggregate_fit(self, rnd, results, failures):
        """Save aggregated model weights and track metrics after each round"""
        aggregation_start = time.time()

        # Calculate bytes received from clients
        bytes_received = 0
        for _, fit_res in results:
            bytes_received += metrics_tracker.calculate_parameters_size(fit_res.parameters)

        # Call parent aggregation (FedAvg)
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        aggregation_time = time.time() - aggregation_start

        if aggregated_weights is not None:
            print(f"\n[Server] Saving round {rnd} aggregated weights (FedAvg)...")
            # Convert parameters to numpy arrays for saving and metrics
            params_to_save = None
            try:
                params_to_save = fl.common.parameters_to_ndarrays(aggregated_weights) if hasattr(aggregated_weights, 'tensors') else aggregated_weights
            except Exception:
                # fallback if conversion fails
                if isinstance(aggregated_weights, list):
                    params_to_save = aggregated_weights
                else:
                    params_to_save = []

            # Save weights file (if we have arrays)
            if isinstance(params_to_save, list) and len(params_to_save) > 0 and isinstance(params_to_save[0], np.ndarray):
                np.savez(f"round-{rnd}-weights_fedavg.npz", *params_to_save)

            # Track metrics - number of parameters (elements) and bytes
            if isinstance(params_to_save, list) and all(isinstance(p, np.ndarray) for p in params_to_save):
                num_parameters_elements = int(sum([p.size for p in params_to_save]))
                num_parameter_bytes = int(sum([p.nbytes for p in params_to_save]))
            else:
                num_parameters_elements = 0
                num_parameter_bytes = 0

            metrics_tracker.add_compute_metrics(rnd, aggregation_time, num_parameters_elements, num_parameter_bytes)

            # Retrieve previously stored bytes_sent (sent at configure_fit), and add communication overhead record
            bytes_sent = metrics_tracker.round_metrics.get(rnd, {}).get('bytes_sent_outgoing', 0)
            metrics_tracker.add_communication_overhead(rnd, bytes_sent, bytes_received, len(results))

        # End round timing and print summary
        metrics_tracker.end_round(rnd)
        metrics_tracker.print_round_summary(rnd)

        return aggregated_weights


class SaveModelFedProxStrategy(BaseRoundRobinStrategy, fl.server.strategy.FedProx):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_fit_index = 0
        self._last_eval_index = 0

    def configure_fit(self, server_round, parameters, client_manager):
        # Start tracking round (timing)
        metrics_tracker.start_round(server_round)

        config = {"server_round": server_round, "proximal_mu": self.proximal_mu}
        available_clients = self.wait_for_clients(client_manager, self.min_available_clients)
        client_ids = list(available_clients.keys())
        num_connected = len(client_ids)

        sample_size, _ = self.num_fit_clients(num_connected)

        selected_clients = []
        for i in range(sample_size):
            index = (self._last_fit_index + i) % num_connected
            selected_clients.append(available_clients[client_ids[index]])
        self._last_fit_index = (self._last_fit_index + sample_size) % num_connected
        selected_ids = [c.cid for c in selected_clients]

        print(f"\n[Server] Round {server_round} - Connected clients: {num_connected}")
        print(f"[Server] Client IDs: {client_ids}")
        print(f"[Server] Sample fraction: {self.fraction_fit} → Selecting {sample_size} clients for training (FedProx)")
        print(f"[Server] Selected clients for training (round-robin): {selected_ids}")

        # Calculate bytes sent (parameters to clients) and store it for later use
        bytes_sent = metrics_tracker.calculate_parameters_size(parameters) * sample_size
        print(f"[Server] Sending {bytes_sent:,} bytes ({bytes_sent/1024/1024:.2f} MB) to {sample_size} clients")
        metrics_tracker.set_bytes_sent_outgoing(server_round, bytes_sent)

        return [(client, fl.common.FitIns(parameters, config)) for client in selected_clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {"server_round": server_round}
        available_clients = self.wait_for_clients(client_manager, self.min_available_clients)
        client_ids = list(available_clients.keys())
        num_connected = len(client_ids)

        sample_size, _ = self.num_evaluation_clients(num_connected)

        selected_clients = []
        for i in range(sample_size):
            index = (self._last_eval_index + i) % num_connected
            selected_clients.append(available_clients[client_ids[index]])
        self._last_eval_index = (self._last_eval_index + sample_size) % num_connected
        selected_ids = [c.cid for c in selected_clients]

        print(f"[Server] Sample fraction: {self.fraction_evaluate} → Selecting {sample_size} clients for evaluation (FedProx)")
        print(f"[Server] Selected clients for evaluation (round-robin): {selected_ids}")

        return [(client, fl.common.EvaluateIns(parameters, config)) for client in selected_clients]

    def aggregate_fit(self, rnd, results, failures):
        """Save aggregated model weights and track metrics after each round (FedProx)"""
        aggregation_start = time.time()

        # Calculate bytes received from clients
        bytes_received = 0
        for _, fit_res in results:
            bytes_received += metrics_tracker.calculate_parameters_size(fit_res.parameters)

        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        aggregation_time = time.time() - aggregation_start

        if aggregated_weights is not None:
            print(f"\n[Server] Saving round {rnd} aggregated weights (FedProx)...")
            # Convert parameters to numpy arrays for saving and metrics
            params_to_save = None
            try:
                params_to_save = fl.common.parameters_to_ndarrays(aggregated_weights) if hasattr(aggregated_weights, 'tensors') else aggregated_weights
            except Exception:
                if isinstance(aggregated_weights, list):
                    params_to_save = aggregated_weights
                else:
                    params_to_save = []

            # Save weights file (if we have arrays)
            if isinstance(params_to_save, list) and len(params_to_save) > 0 and isinstance(params_to_save[0], np.ndarray):
                np.savez(f"round-{rnd}-weights_fedprox.npz", *params_to_save)

            # Track metrics - number of parameters (elements) and bytes
            if isinstance(params_to_save, list) and all(isinstance(p, np.ndarray) for p in params_to_save):
                num_parameters_elements = int(sum([p.size for p in params_to_save]))
                num_parameter_bytes = int(sum([p.nbytes for p in params_to_save]))
            else:
                num_parameters_elements = 0
                num_parameter_bytes = 0

            metrics_tracker.add_compute_metrics(rnd, aggregation_time, num_parameters_elements, num_parameter_bytes)

            # Retrieve previously stored bytes_sent (sent at configure_fit), and add communication overhead record
            bytes_sent = metrics_tracker.round_metrics.get(rnd, {}).get('bytes_sent_outgoing', 0)
            metrics_tracker.add_communication_overhead(rnd, bytes_sent, bytes_received, len(results))

        # End round timing and print summary
        metrics_tracker.end_round(rnd)
        metrics_tracker.print_round_summary(rnd)

        return aggregated_weights


# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_arguments()

    # Display server configuration clearly
    print("\n===== Server Configuration =====")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"IP Address: {args.ip}")
    print(f"Port: {args.port}")
    print(f"Number of Rounds: {args.num_rounds}")
    print(f"Total Clients Expected: {args.total_clients}")
    print(f"Sample Fraction: {args.sample_fraction} → Clients per round: {args.selected_clients}")
    print(f"Minimum Fit Clients: {args.min_fit_clients}")
    print(f"Minimum Evaluate Clients: {args.min_evaluate_clients}")
    if args.strategy == "fedprox":
        print(f"FedProx Mu: {args.proximal_mu}")
    print("=================================\n")

    server_addr = f"{args.ip}:{args.port}"

    # Select strategy
    if args.strategy == "fedavg":
        strategy = SaveModelFedAvgStrategy(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.total_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:  # fedprox
        strategy = SaveModelFedProxStrategy(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.total_clients,
            proximal_mu=args.proximal_mu,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    log_file = "log.txt"
    open(log_file, "w").close()

    # Configure logs
    fl.common.logger.configure(identifier="FL_Test", filename=log_file)

    print(f"Starting server... Waiting for {args.total_clients} clients to connect.")
    print(f"Each round will sample {args.selected_clients} clients for training and evaluation.")

    # Start server
    training_start_time = time.time()
    fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    total_training_time = time.time() - training_start_time

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"{'='*60}\n")

    metrics_tracker.print_final_summary()