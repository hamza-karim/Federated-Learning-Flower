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

    # Minimum number of clients
    try:
        args.min_num_clients = int(input("Enter minimum number of clients required (e.g., 3): "))
    except ValueError:
        print("Invalid input. Using default: 3 clients.")
        args.min_num_clients = 3

    # Sample fraction
    try:
        args.sample_fraction = float(input("Enter fraction of available clients to be used: "))
    except ValueError:
        print("Invalid input. Using default: 1.0")
        args.sample_fraction = 1.0

    # FedProx-specific parameter
    if args.strategy == "fedprox":
        try:
            args.proximal_mu = float(input("Enter FedProx proximal term mu (e.g., 0.1): "))
        except ValueError:
            print("Invalid input. Using default: 0.1")
            args.proximal_mu = 0.1

    return args

# -------------------- Round Configuration --------------------
def get_round_config(server_round: int) -> Dict:
    config = {"server_round": server_round}
    if args.strategy == "fedprox":
        config["proximal_mu"] = args.proximal_mu
    return config

# -------------------- Metrics Aggregation --------------------
def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    mape = [num_examples * m["mape"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"mape": sum(mape) / sum(examples)}

# -------------------- Strategies --------------------
class SaveModelFedAvgStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated weights (FedAvg)...")
            np.savez(f"round-{rnd}-weights_fedavg.npz", *aggregated_weights)
        return aggregated_weights

class SaveModelFedProxStrategy(fl.server.strategy.FedProx):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated weights (FedProx)...")
            np.savez(f"round-{rnd}-weights_fedprox.npz", *aggregated_weights)
        return aggregated_weights

# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_arguments()

    print("\n===== Server Configuration =====")
    print(f"Strategy: {args.strategy}")
    print(f"IP Address: {args.ip}")
    print(f"Port: {args.port}")
    print(f"Number of Rounds: {args.num_rounds}")
    print(f"Sample Fraction: {args.sample_fraction}")
    print(f"Minimum Number of Clients: {args.min_num_clients}")
    if args.strategy == "fedprox":
        print(f"FedProx Mu: {args.proximal_mu}")
    print()
    print("=========================\n")


    server_addr = f"{args.ip}:{args.port}"

    # Select strategy
    if args.strategy == "fedavg":
        strategy = SaveModelFedAvgStrategy(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_num_clients,
            min_evaluate_clients=args.min_num_clients,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=get_round_config,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = SaveModelFedProxStrategy(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_num_clients,
            min_evaluate_clients=args.min_num_clients,
            min_available_clients=args.min_num_clients,
            proximal_mu=args.proximal_mu,
            on_fit_config_fn=get_round_config,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    # Configure logs
    fl.common.logger.configure(identifier="FL_Test", filename="log.txt")

    # Start server
    fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
