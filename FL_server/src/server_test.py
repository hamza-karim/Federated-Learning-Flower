# Muhammad Hamza Karim
# ENHANCED SERVER with Round Completion Breakdown Tracking + DER Network Latency

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

# Import the enhanced tracker
from enhanced_metrics_tracker import EnhancedMetricsTracker

# ============================================
# NEW: Import DER Latency Simulator
# ============================================
from latency_simulator import DERLatencyInjector

faulthandler.enable()

# -------------------- Initialize Enhanced Metrics Tracker --------------------
metrics_tracker = EnhancedMetricsTracker()

# ============================================
# GLOBAL: DER Latency Injector (initialized later)
# ============================================
der_latency = None

# -------------------- Arguments --------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Embedded devices - Enhanced Metrics")
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

    # ============================================
    # NEW: DER Network Latency Configuration
    # ============================================
    print("\n" + "="*60)
    print("DER Network Latency Configuration")
    print("="*60)
    
    latency_choice = input("Enable DER network latency simulation? (y/n): ").strip().lower()
    
    if latency_choice == 'y':
        print("\nLatency Profiles:")
        print("  1. Baseline (0-2ms) - Ideal LAN conditions")
        print("  2. DER Heterogeneous (70% Good: 100ms, 30% Poor: 250ms)")
        
        profile_choice = input("\nSelect profile (1 or 2): ").strip()
        
        if profile_choice == "1":
            args.enable_latency = False
            print("\n✓ Selected: Baseline (no latency injection)")
        else:
            args.enable_latency = True
            
            # Allow customization of good/poor split
            custom = input("\nUse default 70/30 split? (y/n): ").strip().lower()
            if custom == 'n':
                try:
                    good_pct = float(input("Enter % of clients with good connectivity (0-100): "))
                    args.good_connectivity_pct = good_pct / 100.0
                except:
                    args.good_connectivity_pct = 0.7
                    print("Invalid input, using default 70%")
            else:
                args.good_connectivity_pct = 0.7
            
            print(f"\n✓ Selected: DER Heterogeneous Network")
            print(f"  - Good Connectivity: {args.good_connectivity_pct*100:.0f}% (100ms ± 25ms)")
            print(f"  - Poor Connectivity: {(1-args.good_connectivity_pct)*100:.0f}% (250ms ± 25ms)")
    else:
        args.enable_latency = False
        print("\n✓ Latency simulation DISABLED")
    
    print("="*60 + "\n")

    return args

# --- Aggregation functions ---
def weighted_average_fit(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    """Aggregates train_loss from clients during fit()"""
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"train_loss": sum(losses) / sum(examples)}

def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    """Aggregates MAPE from clients during evaluate()"""
    mape = [num_examples * m["mape"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"mape": sum(mape) / sum(examples)}

# -------------------- BASE STRATEGY WITH ID MAPPING --------------------
class BaseRoundRobinStrategy:
    def __init__(self):
        self.client_id_map = {}

    def resolve_client_ids(self, available_clients):
        """Query clients for their ID and update the map."""
        for cid, client_proxy in available_clients.items():
            if cid not in self.client_id_map:
                try:
                    ins = fl.common.GetPropertiesIns(config={})
                    res = client_proxy.get_properties(ins, timeout=5)
                    logical_id = res.properties["client_id"]
                    self.client_id_map[cid] = int(logical_id)
                    print(f"[Server] ✅ Registered {cid} as Client {logical_id}")
                except Exception:
                    pass

        sorted_clients = sorted(
            available_clients.items(),
            key=lambda item: self.client_id_map.get(item[0], 9999)
        )
        return sorted_clients

    def wait_for_clients(self, client_manager, min_clients):
        available_clients = client_manager.all()

        while len(available_clients) < min_clients:
            self.resolve_client_ids(available_clients)

            connected_ids = set(self.client_id_map.values())
            expected_ids = set(range(1, min_clients + 1))
            missing_ids = sorted(list(expected_ids - connected_ids))

            print(f"\n[Server] Status: {len(available_clients)}/{min_clients} connected.")
            if missing_ids:
                print(f"[Server] ⚠️  MISSING CLIENTS: {missing_ids}")
            else:
                print(f"[Server] ⚠️  Resolving IDs for {len(available_clients) - len(connected_ids)} clients...")

            time.sleep(5)
            available_clients = client_manager.all()

        return available_clients

# -------------------- ENHANCED FEDAVG WITH DETAILED TRACKING --------------------
class SaveModelFedAvgStrategy(BaseRoundRobinStrategy, fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        BaseRoundRobinStrategy.__init__(self)
        fl.server.strategy.FedAvg.__init__(self, **kwargs)
        self._last_fit_index = 0
        self._last_eval_index = 0
        self.client_fit_dispatch_times = {}

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
            
            logical_id = self.client_id_map.get(cid, "Unknown")
            selected_logical_ids.append(logical_id)
            
            client_id_str = str(logical_id)
            self.client_fit_dispatch_times[client_id_str] = time.time()
            
            # ============================================
            # NEW: Inject DER network latency
            # ============================================
            if der_latency is not None:
                latency_ms = der_latency.inject_latency(
                    client_id=int(client_id_str), 
                    total_clients=self.min_available_clients,
                    round_num=server_round
                )
                
                # Record latency in metrics tracker
                if latency_ms > 0:
                    metrics_tracker.record_network_latency(server_round, client_id_str, latency_ms)
            
            # Record fit start time
            metrics_tracker.record_client_fit_start(server_round, client_id_str)

        self._last_fit_index = (self._last_fit_index + sample_size) % num_connected

        print(f"\n[Server] Round {server_round} - Connected clients: {num_connected}")
        print(f"[Server] Selecting {sample_size} clients: {selected_logical_ids}")
        
        # ============================================
        # NEW: Print latency summary for this round
        # ============================================
        if der_latency is not None and der_latency.enable_latency:
            der_latency.print_round_summary(server_round)

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
        for client_proxy, fit_res in results:
            try:
                client_id = fit_res.metrics.get("client_id", None)

                if client_id is not None:
                    client_id_str = str(client_id)

                    metrics_tracker.record_client_fit_end(rnd, client_id_str)
                    metrics_tracker.record_client_upload_start(rnd, client_id_str)
                    metrics_tracker.record_client_upload_end(rnd, client_id_str)

                    # ============================================
                    # CRITICAL FIX FOR PAPER METRICS
                    # ============================================
                    # Use client's reported "Pure Compute Time" to overwrite server timestamp
                    # This ensures "Network Overhead" = Round_Time - Pure_Compute
                    reported_duration = fit_res.metrics.get("training_duration", 0)
                    
                    if reported_duration > 0:
                        print(f"[Server] Client {client_id} reported training time: {reported_duration:.2f}s")
                        
                        # OVERWRITE: Back-calculate start time based on reported duration
                        if client_id_str in metrics_tracker.client_fit_end_times[rnd]:
                            end_time = metrics_tracker.client_fit_end_times[rnd][client_id_str]
                            metrics_tracker.client_fit_start_times[rnd][client_id_str] = end_time - reported_duration
                    else:
                        print(f"[Warning] Client {client_id} reported 0s duration")
                else:
                    print(f"[Warning] Client didn't report ID in metrics - can't track timing")

            except Exception as e:
                print(f"[Warning] Error extracting client metrics: {e}")

        metrics_tracker.record_aggregation_start(rnd)
        aggregation_start = time.time()

        bytes_received = sum([metrics_tracker.calculate_parameters_size(fit_res.parameters)
                             for _, fit_res in results])

        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        aggregation_time = time.time() - aggregation_start
        metrics_tracker.record_aggregation_end(rnd)

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


# -------------------- ENHANCED FEDPROX --------------------
class SaveModelFedProxStrategy(BaseRoundRobinStrategy, fl.server.strategy.FedProx):
    def __init__(self, **kwargs):
        BaseRoundRobinStrategy.__init__(self)
        fl.server.strategy.FedProx.__init__(self, **kwargs)
        self._last_fit_index = 0
        self._last_eval_index = 0
        self.client_fit_dispatch_times = {}

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
            
            logical_id = self.client_id_map.get(cid, "Unknown")
            selected_logical_ids.append(logical_id)
            
            client_id_str = str(logical_id)
            self.client_fit_dispatch_times[client_id_str] = time.time()
            
            # ============================================
            # NEW: Inject DER network latency
            # ============================================
            if der_latency is not None:
                latency_ms = der_latency.inject_latency(
                    client_id=int(client_id_str), 
                    total_clients=self.min_available_clients,
                    round_num=server_round
                )
                
                if latency_ms > 0:
                    metrics_tracker.record_network_latency(server_round, client_id_str, latency_ms)
            
            metrics_tracker.record_client_fit_start(server_round, client_id_str)

        self._last_fit_index = (self._last_fit_index + sample_size) % num_connected

        print(f"\n[Server] Round {server_round} - Connected clients: {num_connected}")
        print(f"[Server] Selecting {sample_size} clients: {selected_logical_ids}")
        
        # ============================================
        # NEW: Print latency summary
        # ============================================
        if der_latency is not None and der_latency.enable_latency:
            der_latency.print_round_summary(server_round)

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
        for client_proxy, fit_res in results:
            try:
                client_id = fit_res.metrics.get("client_id", None)

                if client_id is not None:
                    client_id_str = str(client_id)

                    metrics_tracker.record_client_fit_end(rnd, client_id_str)
                    metrics_tracker.record_client_upload_start(rnd, client_id_str)
                    metrics_tracker.record_client_upload_end(rnd, client_id_str)

                    # ============================================
                    # CRITICAL FIX FOR PAPER METRICS
                    # ============================================
                    reported_duration = fit_res.metrics.get("training_duration", 0)
                    
                    if reported_duration > 0:
                        print(f"[Server] Client {client_id} reported training time: {reported_duration:.2f}s")
                        
                        # OVERWRITE: Back-calculate start time based on reported duration
                        if client_id_str in metrics_tracker.client_fit_end_times[rnd]:
                            end_time = metrics_tracker.client_fit_end_times[rnd][client_id_str]
                            metrics_tracker.client_fit_start_times[rnd][client_id_str] = end_time - reported_duration
                    else:
                        print(f"[Warning] Client {client_id} reported 0s duration")

                else:
                    print(f"[Warning] Client didn't report ID in metrics - can't track timing")

            except Exception as e:
                print(f"[Warning] Error extracting client metrics: {e}")

        metrics_tracker.record_aggregation_start(rnd)
        aggregation_start = time.time()

        bytes_received = sum([metrics_tracker.calculate_parameters_size(fit_res.parameters)
                             for _, fit_res in results])

        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        aggregation_time = time.time() - aggregation_start
        metrics_tracker.record_aggregation_end(rnd)

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
    
    # ============================================
    # NEW: Initialize DER Latency Injector
    # ============================================
    if args.enable_latency:
        der_latency = DERLatencyInjector(
            good_connectivity_percentage=getattr(args, 'good_connectivity_pct', 0.7),
            enable_latency=True,
            seed=42,
            verbose=True
        )
    else:
        der_latency = DERLatencyInjector(
            enable_latency=False,
            seed=42,
            verbose=True
        )

    print("\n===== Enhanced Server Configuration =====")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"IP: {args.ip}:{args.port}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Clients: {args.total_clients}")
    print(f"Enhanced Metrics: ENABLED")
    print(f"DER Latency: {'ENABLED' if args.enable_latency else 'DISABLED'}")
    print("=========================================\n")

    if args.strategy == "fedavg":
        strategy = SaveModelFedAvgStrategy(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.total_clients,
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
            fit_metrics_aggregation_fn=weighted_average_fit,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    log_file = "log.txt"
    open(log_file, "w").close()
    fl.common.logger.configure(identifier="FL_Test", filename=log_file)

    print(f"Starting enhanced server... Waiting for {args.total_clients} clients.")
    training_start_time = time.time()

    fl.server.start_server(
        server_address=f"{args.ip}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    # Print comprehensive final summary
    metrics_tracker.print_final_summary()
    
    # ============================================
    # NEW: Export latency statistics if enabled
    # ============================================
    if der_latency is not None and der_latency.enable_latency:
        print("\n" + "="*70)
        print("Exporting DER Network Latency Statistics...")
        print("="*70)
        
        latency_stats = der_latency.export_statistics()
        
        with open('der_latency_stats.json', 'w') as f:
            json.dump(latency_stats, f, indent=2)
        
        print("✅ Latency statistics saved to: der_latency_stats.json")
        print("="*70 + "\n")