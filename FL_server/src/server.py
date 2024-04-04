import argparse
import flwr as fl
from typing import List, Tuple
import time

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--ip",
    help="Provide the IP address",
    default="0.0.0.0",
    required=False,
)
parser.add_argument(
    "--port",
    help="Provide the Port address",
    default="8080",
    required=False,
)
parser.add_argument(
    "--num_rounds",
    help="Number of rounds of federated learning (default: 5)",
    type=int,
    default=5,
    required=False,
)
parser.add_argument(
    "--sample_fraction",
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    type=float,
    default=1.0,
    required=False,
)
parser.add_argument(
    "--min_num_clients",
    help="Minimum number of available clients required for sampling (default: 2)",
    type=int,
    default=2,
    required=False,
)


def weighted_average(metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:
    accuracies = [num_examples * m["mape"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"mape": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    config = {
        "epochs": 3,
        "batch_size": 16,
    }
    return config


def main():
    args = parser.parse_args()

    server_addr = f"{args.ip}:{args.port}"
    num_rounds = args.num_rounds
    sample_fraction = args.sample_fraction
    min_num_clients = args.min_num_clients

    print(f"server addr - {server_addr}")

    start_time = time.time()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=sample_fraction,
        fraction_evaluate=sample_fraction,
        min_fit_clients=min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print("--------------------")
    print(f"START TIME - {start_time}")

    fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    end_time = time.time()
    print("--------------------")
    print(f"END TIME - {end_time}")

    print(f"ELAPSED TIME -  {end_time - start_time}")


if __name__ == "__main__":
    main()