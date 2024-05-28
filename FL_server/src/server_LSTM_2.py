import argparse
from typing import Dict
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Embedded devices")
    parser.add_argument("--ip", help="Provide the IP address", default="0.0.0.0", required=False)
    parser.add_argument("--port", help="Provide the Port address", default="8080", required=False)
    parser.add_argument("--num_rounds", help="Number of rounds of federated learning (default: 5)", type=int, default=5, required=False)
    parser.add_argument("--sample_fraction", help="Fraction of available clients used for fit/evaluate (default: 1.0)", type=float, default=1.0, required=False)
    parser.add_argument("--min_num_clients", help="Minimum number of available clients required for sampling (default: 2)", type=int, default=2, required=False)
    return parser.parse_args()

# Current round configuration
def get_round_config(server_round: int) -> Dict:
    return {"server_round": server_round}


def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    accuracies = [num_examples * m["mape"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"mape": sum(accuracies) / sum(examples)}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

if __name__ == "__main__":
    args = parse_arguments()
    print("Server Configuration:")
    print(f"IP Address: {args.ip}")
    print(f"Port: {args.port}")
    print(f"Number of Rounds: {args.num_rounds}")
    print(f"Sample Fraction: {args.sample_fraction}")
    print(f"Minimum Number of Clients: {args.min_num_clients}")

    server_addr = f"{args.ip}:{args.port}"
    num_rounds = args.num_rounds
    sample_fraction = args.sample_fraction
    min_num_clients = args.min_num_clients


    # # Create strategy and run server
    # strategy = SaveModelStrategy()

    # Building Strategy
    strategy = SaveModelStrategy(
        fraction_fit=sample_fraction,
        fraction_evaluate=sample_fraction,
        min_fit_clients=min_num_clients,
        min_evaluate_clients=min_num_clients,
        min_available_clients=min_num_clients,
        on_fit_config_fn=get_round_config,
        evaluate_metrics_aggregation_fn=weighted_average,
       # fit_metrics_aggregation_fn=weighted_average,
    )

    # Server logs
    fl.common.logger.configure(identifier="FL_Test", filename="log.txt")

    # Start the server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        server_address=server_addr,
    )