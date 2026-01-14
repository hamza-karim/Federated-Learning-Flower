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
from typing import Dict, List, Tuple, Optional
import time
import sys
import json
from collections import defaultdict
faulthandler.enable()

# -------------------- Metrics Tracker (No Changes) --------------------
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
        total_size = 0
        if parameters is None:
            return 0
        if hasattr(parameters, "tensors"):
            try:
                for tensor in parameters.tensors:
                    if isinstance(tensor, (bytes, bytearray)):
                        total_size += len(tensor)
                    else:
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
        self.round_metrics[round_num]['bytes_sent_outgoing'] = int(bytes_sent)

    def add_communication_overhead(self, round_num: int, bytes_sent: int, bytes_received: int, num_clients: int):
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
        self.round_metrics[round_num]['num_parameters'] = int(num_parameters)
        self.round_metrics[round_num]['num_parameter_bytes'] = int(num_parameter_bytes)

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

        self.save_metrics_to_file()

    def save_metrics_to_file(self):
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

metrics_tracker = MetricsTracker()

# -------------------- Arguments --------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Embedded devices - Choose FL strategy")
    parser.add_argument("--ip", help="Provide the IP address", default="0.0.0.0", required=False)
    parser.add_argument("--port", help="Provide the Port address", default="8080", required=False)
    args = parser.parse_args()

    strategy_choice = input("Choose FL strategy (1=FedAvg, 2=FedProx): ").strip()
    args.strategy = "fedavg" if strategy_choice == "1" else "fedprox"

    try:
        args.num_rounds = int(input("Enter the number of federated learning rounds (e.g., 5): "))
    except Exception:
        args.num_rounds = 5

    print("\nClient Configuration Options: 1=4, 2=8, 3=12, 4=16, 5=20, 6=Custom")
    client_choice = input("Choose total number of clients (1-6): ").strip()
    client_options = {"1": 4, "2": 8, "3": 12, "4": 16, "5": 20}
    
    if client_choice in client_options:
        args.total_clients = client_options[client_choice]
    elif client_choice == "6":
        try:
            args.total_clients = int(input("Enter custom number of clients: "))
        except:
            args.total_clients = 4
    else:
        args.total_clients = 4

    try:
        args.sample_fraction = float(input(f"Enter fraction of clients to sample (0.0-1.0): "))
    except:
        args.sample_fraction = 0.5

    args.selected_clients = max(1, int(args.total_clients * args.sample_fraction))
    args.min_fit_clients = args.selected_clients
    args.min_evaluate_clients = args.selected_clients

    if args.strategy == "fedprox":
        try:
            args.proximal_mu = float(input("Enter FedProx proximal term mu (e.g., 0.1): "))
        except:
            args.proximal_mu = 0.1

    return args

# --- NEW: Aggregation function for training metrics (loss) ---
def weighted_average_fit(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    # Aggregates the 'train_loss' returned by clients during fit()
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"train_loss": sum(losses) / sum(examples)}

# --- EXISTING: Aggregation function for evaluation metrics (mape) ---
def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    mape = [num_examples * m["mape"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"mape": sum(mape) / sum(examples)}

# -------------------- BASE STRATEGY WITH ID MAPPING --------------------
class BaseRoundRobinStrategy:
    def __init__(self):
        # Maps IP (cid) to Client ID (e.g., 1, 2)
        self.client_id_map = {}

    def wait_for_clients(self, client_manager, min_clients):
        available_clients = client_manager.all()
        while len(available_clients) < min_clients:
            print(f"[Server] Waiting for clients to connect... {len(available_clients)}/{min_clients}")
            time.sleep(2)
            available_clients = client_manager.all()
        return available_clients

    def resolve_client_ids(self, available_clients):
        """Query clients for their ID and return a SORTED list."""
        for cid, client_proxy in available_clients.items():
            if cid not in self.client_id_map:
                try:
                    # Ask the client for its ID
                    ins = fl.common.GetPropertiesIns(config={})
                    res = client_proxy.get_properties(ins, timeout=30)
                    logical_id = res.properties["client_id"]
                    self.client_id_map[cid] = int(logical_id)
                    print(f"[Server] Registered {cid} as Client {logical_id}")
                except Exception as e:
                    print(f"[Server] Could not query properties for {cid}: {e}")
                    self.client_id_map[cid] = 9999 

        # Sort clients by ID
        sorted_clients = sorted(
            available_clients.items(), 
            key=lambda item: self.client_id_map.get(item[0], 9999)
        )
        return sorted_clients

# -------------------- FEDAVG --------------------
class SaveModelFedAvgStrategy(BaseRoundRobinStrategy, fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        BaseRoundRobinStrategy.__init__(self)
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        self._last_fit_index = 0
        self._last_eval_index = 0

    def configure_fit(self, server_round, parameters, client_manager):
        metrics_tracker.start_round(server_round)
        config = {"server_round": server_round}
        
        available_clients_dict = self.wait_for_clients(client_manager, self.min_available_clients)
        sorted_clients_list = self.resolve_client_ids(available_clients_dict)
        num_connected = len(sorted_clients_list)
        sample_size, _ = self.num_fit_clients(num_connected)

        selected_clients = []
        selected_logical_ids = []
        
        for i in range(sample_size):
            index = (self._last_fit_index + i) % num_connected
            cid, client_proxy = sorted_clients_list[index]
            selected_clients.append(client_proxy)
            selected_logical_ids.append(self.client_id_map.get(cid, "Unknown"))
            
        self._last_fit_index = (self._last_fit_index + sample_size) % num_connected

        print(f"\n[Server] Round {server_round} - Connected clients: {num_connected}")
        print(f"[Server] Selecting {sample_size} clients: {selected_logical_ids}")

        bytes_sent = metrics_tracker.calculate_parameters_size(parameters) * sample_size
        metrics_tracker.set_bytes_sent_outgoing(server_round, bytes_sent)

        return [(client, fl.common.FitIns(parameters, config)) for client in selected_clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {"server_round": server_round}
        available_clients_dict = self.wait_for_clients(client_manager, self.min_available_clients)
        sorted_clients_list = self.resolve_client_ids(available_clients_dict)
        num_connected = len(sorted_clients_list)
        sample_size, _ = self.num_evaluation_clients(num_connected)

        selected_clients = []
        for i in range(sample_size):
            index = (self._last_eval_index + i) % num_connected
            cid, client_proxy = sorted_clients_list[index]
            selected_clients.append(client_proxy)
        self._last_eval_index = (self._last_eval_index + sample_size) % num_connected

        return [(client, fl.common.EvaluateIns(parameters, config)) for client in selected_clients]

    def aggregate_fit(self, rnd, results, failures):
        aggregation_start = time.time()
        bytes_received = sum([metrics_tracker.calculate_parameters_size(fit_res.parameters) for _, fit_res in results])
        
        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        aggregation_time = time.time() - aggregation_start
        if aggregated_weights is not None:
            print(f"\n[Server] Saving round {rnd} aggregated weights...")
            try:
                params_to_save = fl.common.parameters_to_ndarrays(aggregated_weights)
                np.savez(f"round-{rnd}-weights_fedavg.npz", *params_to_save)
                
                num_elements = sum([p.size for p in params_to_save])
                num_bytes = sum([p.nbytes for p in params_to_save])
                metrics_tracker.add_compute_metrics(rnd, aggregation_time, num_elements, num_bytes)
            except Exception as e:
                print(f"Error saving model: {e}")

            bytes_sent = metrics_tracker.round_metrics.get(rnd, {}).get('bytes_sent_outgoing', 0)
            metrics_tracker.add_communication_overhead(rnd, bytes_sent, bytes_received, len(results))

        metrics_tracker.end_round(rnd)
        metrics_tracker.print_round_summary(rnd)
        return aggregated_weights, aggregated_metrics

# -------------------- FEDPROX --------------------
class SaveModelFedProxStrategy(BaseRoundRobinStrategy, fl.server.strategy.FedProx):
    def __init__(self, **kwargs):
        BaseRoundRobinStrategy.__init__(self)
        fl.server.strategy.FedProx.__init__(self, **kwargs)
        self._last_fit_index = 0
        self._last_eval_index = 0

    def configure_fit(self, server_round, parameters, client_manager):
        metrics_tracker.start_round(server_round)
        config = {"server_round": server_round, "proximal_mu": self.proximal_mu}
        
        available_clients_dict = self.wait_for_clients(client_manager, self.min_available_clients)
        sorted_clients_list = self.resolve_client_ids(available_clients_dict)
        num_connected = len(sorted_clients_list)
        sample_size, _ = self.num_fit_clients(num_connected)

        selected_clients = []
        selected_logical_ids = []
        for i in range(sample_size):
            index = (self._last_fit_index + i) % num_connected
            cid, client_proxy = sorted_clients_list[index]
            selected_clients.append(client_proxy)
            selected_logical_ids.append(self.client_id_map.get(cid, "Unknown"))
        self._last_fit_index = (self._last_fit_index + sample_size) % num_connected

        print(f"\n[Server] Round {server_round} - Connected clients: {num_connected}")
        print(f"[Server] Selecting {sample_size} clients: {selected_logical_ids}")

        bytes_sent = metrics_tracker.calculate_parameters_size(parameters) * sample_size
        metrics_tracker.set_bytes_sent_outgoing(server_round, bytes_sent)
        return [(client, fl.common.FitIns(parameters, config)) for client in selected_clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {"server_round": server_round}
        available_clients_dict = self.wait_for_clients(client_manager, self.min_available_clients)
        sorted_clients_list = self.resolve_client_ids(available_clients_dict)
        num_connected = len(sorted_clients_list)
        sample_size, _ = self.num_evaluation_clients(num_connected)
        
        selected_clients = []
        for i in range(sample_size):
            index = (self._last_eval_index + i) % num_connected
            cid, client_proxy = sorted_clients_list[index]
            selected_clients.append(client_proxy)
        self._last_eval_index = (self._last_eval_index + sample_size) % num_connected
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in selected_clients]

    def aggregate_fit(self, rnd, results, failures):
        aggregation_start = time.time()
        bytes_received = sum([metrics_tracker.calculate_parameters_size(fit_res.parameters) for _, fit_res in results])

        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        aggregation_time = time.time() - aggregation_start
        if aggregated_weights is not None:
            print(f"\n[Server] Saving round {rnd} aggregated weights...")
            try:
                params_to_save = fl.common.parameters_to_ndarrays(aggregated_weights)
                np.savez(f"round-{rnd}-weights_fedprox.npz", *params_to_save)
                
                num_elements = sum([p.size for p in params_to_save])
                num_bytes = sum([p.nbytes for p in params_to_save])
                metrics_tracker.add_compute_metrics(rnd, aggregation_time, num_elements, num_bytes)
            except Exception as e:
                print(f"Error saving model: {e}")

            bytes_sent = metrics_tracker.round_metrics.get(rnd, {}).get('bytes_sent_outgoing', 0)
            metrics_tracker.add_communication_overhead(rnd, bytes_sent, bytes_received, len(results))

        metrics_tracker.end_round(rnd)
        metrics_tracker.print_round_summary(rnd)
        return aggregated_weights, aggregated_metrics

# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_arguments()
    print("\n===== Server Configuration =====")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"IP: {args.ip}:{args.port}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Clients: {args.total_clients}")
    print("=================================\n")

    if args.strategy == "fedavg":
        strategy = SaveModelFedAvgStrategy(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.total_clients,
            # Updated: Added fit metrics aggregation
            fit_metrics_aggregation_fn=weighted_average_fit,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = SaveModelFedProxStrategy(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.total_clients,
            proximal_mu=args.proximal_mu,
            # Updated: Added fit metrics aggregation
            fit_metrics_aggregation_fn=weighted_average_fit,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    log_file = "log.txt"
    open(log_file, "w").close()
    fl.common.logger.configure(identifier="FL_Test", filename=log_file)

    print(f"Starting server... Waiting for {args.total_clients} clients.")
    training_start_time = time.time()
    fl.server.start_server(
        server_address=f"{args.ip}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    metrics_tracker.print_final_summary()