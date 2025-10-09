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

    # Clients per round (instead of sample fraction)
    print(f"\nWith {args.total_clients} total clients:")
    try:
        args.clients_per_round = int(input(f"Enter clients per round (1-{args.total_clients}): "))
        if args.clients_per_round < 1 or args.clients_per_round > args.total_clients:
            raise ValueError()
    except ValueError:
        print("Invalid input. Using 4 clients per round.")
        args.clients_per_round = 4

    # Calculate rounds per cycle and other metrics
    args.rounds_per_cycle = (args.total_clients + args.clients_per_round - 1) // args.clients_per_round
    args.total_cycles = (args.num_rounds + args.rounds_per_cycle - 1) // args.rounds_per_cycle
    
    print(f"\nConfiguration Summary:")
    print(f"Total clients: {args.total_clients}")
    print(f"Clients per round: {args.clients_per_round}")
    print(f"Rounds per cycle: {args.rounds_per_cycle}")
    print(f"Total rounds: {args.num_rounds}")
    print(f"Complete cycles: {args.total_cycles}")

    # Print the schedule
    print(f"\nRound-Robin Schedule:")
    for round_num in range(1, min(args.num_rounds + 1, args.rounds_per_cycle * 2 + 1)):  # Show first 2 cycles
        start_client = ((round_num - 1) % args.rounds_per_cycle) * args.clients_per_round + 1
        end_client = min(start_client + args.clients_per_round - 1, args.total_clients)
        cycle_num = ((round_num - 1) // args.rounds_per_cycle) + 1
        print(f"Round {round_num} (Cycle {cycle_num}): Clients {start_client}-{end_client}")
        
        if round_num == args.rounds_per_cycle * 2 and args.num_rounds > round_num:
            print("...")
            break

    # Set minimum clients to be the same as clients per round for round-robin
    args.min_fit_clients = args.clients_per_round
    args.min_evaluate_clients = args.clients_per_round

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

# -------------------- Custom Evaluation Function --------------------
def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters, config):
        print(f"Server-side evaluation round {server_round}")
        # Add your server-side evaluation logic here if needed
        return None  # Return (loss, metrics) if implementing server-side eval
    return evaluate

# -------------------- Round-Robin Client Selector --------------------
class RoundRobinClientSelector:
    def __init__(self, total_clients: int, clients_per_round: int):
        self.total_clients = total_clients
        self.clients_per_round = clients_per_round
        self.rounds_per_cycle = (total_clients + clients_per_round - 1) // clients_per_round
        
    def get_client_ids_for_round(self, server_round: int) -> List[str]:
        """Get the specific client IDs for this round based on round-robin scheduling."""
        # Convert to 0-based indexing for calculations
        round_in_cycle = (server_round - 1) % self.rounds_per_cycle
        
        # Calculate which client indices should be used
        start_idx = round_in_cycle * self.clients_per_round
        end_idx = min(start_idx + self.clients_per_round, self.total_clients)
        
        # Create a list of client IDs that should be used (1-based)
        target_client_ids = [str(i) for i in range(start_idx + 1, end_idx + 1)]
        
        return target_client_ids

# -------------------- Strategies --------------------
class SaveModelFedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(self, client_selector: RoundRobinClientSelector, **kwargs):
        super().__init__(**kwargs)
        self.round_info = []
        self.client_selector = client_selector
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training - ROUND-ROBIN SELECTION."""
        config = {"server_round": server_round}
        if hasattr(args, 'proximal_mu'):  # For consistency
            config["proximal_mu"] = getattr(args, 'proximal_mu', 0.0)
        
        # Get target client IDs for this round
        target_client_ids = self.client_selector.get_client_ids_for_round(server_round)
        
        # Get all available clients and manually filter for round-robin
        all_available_cids = client_manager.all()
        selected_cids = [cid for cid in all_available_cids if cid in target_client_ids]
        
        # Get the actual client objects
        clients = [client_manager.clients[cid] for cid in selected_cids]
        
        # Log which clients were selected this round
        cycle_num = ((server_round - 1) // self.client_selector.rounds_per_cycle) + 1
        round_in_cycle = ((server_round - 1) % self.client_selector.rounds_per_cycle) + 1
        
        print(f"Round {server_round} (Cycle {cycle_num}, Round {round_in_cycle}): Selected {len(clients)} clients {selected_cids}")
        
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation - ROUND-ROBIN SELECTION."""
        config = {"server_round": server_round}
        
        # Get target client IDs for this round
        target_client_ids = self.client_selector.get_client_ids_for_round(server_round)
        
        # Get all available clients and manually filter for round-robin
        all_available_cids = client_manager.all()
        selected_cids = [cid for cid in all_available_cids if cid in target_client_ids]
        
        # Get the actual client objects
        clients = [client_manager.clients[cid] for cid in selected_cids]
        
        # Log which clients were selected for evaluation this round
        print(f"Round {server_round}: Evaluating on {len(clients)} clients {selected_cids}")
        
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {server_round} aggregated weights (FedAvg)...")
            # Extract parameters from the tuple (parameters, metrics)
            parameters, metrics = aggregated_weights
            # Convert Parameters object to numpy arrays
            weights_as_arrays = fl.common.parameters_to_ndarrays(parameters)
            np.savez(f"round-{server_round}-weights_fedavg.npz", *weights_as_arrays)
            
            # Store round information
            self.round_info.append({
                'round': server_round,
                'participating_clients': len(results),
                'strategy': 'fedavg'
            })
        return aggregated_weights

class SaveModelFedProxStrategy(fl.server.strategy.FedProx):
    def __init__(self, client_selector: RoundRobinClientSelector, **kwargs):
        super().__init__(**kwargs)
        self.round_info = []
        self.client_selector = client_selector
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training - ROUND-ROBIN SELECTION."""
        config = {"server_round": server_round, "proximal_mu": self.proximal_mu}
        
        # Get target client IDs for this round
        target_client_ids = self.client_selector.get_client_ids_for_round(server_round)
        
        # Get all available clients and manually filter for round-robin
        all_available_cids = client_manager.all()
        selected_cids = [cid for cid in all_available_cids if cid in target_client_ids]
        
        # Get the actual client objects
        clients = [client_manager.clients[cid] for cid in selected_cids]
        
        # Log which clients were selected this round
        cycle_num = ((server_round - 1) // self.client_selector.rounds_per_cycle) + 1
        round_in_cycle = ((server_round - 1) % self.client_selector.rounds_per_cycle) + 1
        
        print(f"Round {server_round} (Cycle {cycle_num}, Round {round_in_cycle}): Selected {len(clients)} clients {selected_cids}")
        
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation - ROUND-ROBIN SELECTION."""
        config = {"server_round": server_round}
        
        # Get all available clients
        available_clients = list(client_manager._clients.values())
        
        # Use the same round-robin selection for evaluation
        clients = self.client_selector.get_clients_for_round(server_round, available_clients)
        
        # Log which clients were selected for evaluation this round
        client_ids = [client.cid for client in clients]
        print(f"Round {server_round}: Evaluating on {len(clients)} clients {client_ids}")
        
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {server_round} aggregated weights (FedProx)...")
            # Extract parameters from the tuple (parameters, metrics)
            parameters, metrics = aggregated_weights
            # Convert Parameters object to numpy arrays
            weights_as_arrays = fl.common.parameters_to_ndarrays(parameters)
            np.savez(f"round-{server_round}-weights_fedprox.npz", *weights_as_arrays)
            
            # Store round information
            self.round_info.append({
                'round': server_round,
                'participating_clients': len(results),
                'strategy': 'fedprox',
                'proximal_mu': self.proximal_mu
            })
        return aggregated_weights

# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_arguments()

    print("\n===== Server Configuration =====")
    print(f"Strategy: {args.strategy}")
    print(f"IP Address: {args.ip}")
    print(f"Port: {args.port}")
    print(f"Number of Rounds: {args.num_rounds}")
    print(f"Total Clients: {args.total_clients}")
    print(f"Clients per Round: {args.clients_per_round}")
    print(f"Rounds per Cycle: {args.rounds_per_cycle}")
    print(f"Selection Mode: Round-Robin (Sequential)")
    if args.strategy == "fedprox":
        print(f"FedProx Mu: {args.proximal_mu}")
    print("=================================\n")

    server_addr = f"{args.ip}:{args.port}"

    # Create the round-robin client selector
    client_selector = RoundRobinClientSelector(args.total_clients, args.clients_per_round)

    # Select strategy with proper configuration and round-robin selector
    if args.strategy == "fedavg":
        strategy = SaveModelFedAvgStrategy(
            client_selector=client_selector,
            fraction_fit=1.0,                            # Not used in round-robin, but set to 1.0
            fraction_evaluate=1.0,                       # Not used in round-robin, but set to 1.0
            min_fit_clients=args.min_fit_clients,        # Minimum clients needed for training
            min_evaluate_clients=args.min_evaluate_clients, # Minimum clients needed for evaluation
            min_available_clients=args.total_clients,    # Wait for this many total clients
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:  # fedprox
        strategy = SaveModelFedProxStrategy(
            client_selector=client_selector,
            fraction_fit=1.0,                            # Not used in round-robin, but set to 1.0
            fraction_evaluate=1.0,                       # Not used in round-robin, but set to 1.0
            min_fit_clients=args.min_fit_clients,        # Minimum clients needed for training
            min_evaluate_clients=args.min_evaluate_clients, # Minimum clients needed for evaluation
            min_available_clients=args.total_clients,    # Wait for this many total clients
            proximal_mu=args.proximal_mu,                # FedProx parameter
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    # Configure logs
    fl.common.logger.configure(identifier="FL_Test", filename="log.txt")

    print(f"Starting server... Waiting for {args.total_clients} clients to connect.")
    print(f"Using Round-Robin selection: {args.clients_per_round} clients per round in {args.rounds_per_cycle}-round cycles.")

    # Start server
    fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )