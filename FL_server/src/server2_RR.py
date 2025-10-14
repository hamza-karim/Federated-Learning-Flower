# Muhammad Hamza Karim

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "default"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import argparse
import flwr as fl
import numpy as np
import faulthandler
from typing import Dict, List, Tuple
import time
faulthandler.enable()  

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
    except ValueError:
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
        except ValueError:
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
    except ValueError:
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
    except ValueError:
        args.min_fit_clients = args.selected_clients
        args.min_evaluate_clients = args.selected_clients

    # FedProx-specific parameter
    if args.strategy == "fedprox":
        try:
            args.proximal_mu = float(input("Enter FedProx proximal term mu (e.g., 0.1): "))
        except ValueError:
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
        # Add server-side evaluation logic if needed
        return None
    return evaluate

# -------------------- Strategies with Round-Robin --------------------
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
        self.round_info = []
        self._last_fit_index = 0
        self._last_eval_index = 0

    def configure_fit(self, server_round, parameters, client_manager):
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
        """Save aggregated model weights after each round"""
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"\n[Server] Saving round {rnd} aggregated weights (FedAvg)...")
            np.savez(f"round-{rnd}-weights_fedavg.npz", *aggregated_weights)
        return aggregated_weights


class SaveModelFedProxStrategy(BaseRoundRobinStrategy, fl.server.strategy.FedProx):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_info = []
        self._last_fit_index = 0
        self._last_eval_index = 0

    def configure_fit(self, server_round, parameters, client_manager):
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
        """Save aggregated model weights after each round"""
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"\n[Server] Saving round {rnd} aggregated weights (FedProx)...")
            np.savez(f"round-{rnd}-weights_fedprox.npz", *aggregated_weights)
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

    # Configure logs
    fl.common.logger.configure(identifier="FL_Test", filename="log.txt")

    print(f"Starting server... Waiting for {args.total_clients} clients to connect.")
    print(f"Each round will sample {args.selected_clients} clients for training and evaluation.")

    # Start server
    fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )