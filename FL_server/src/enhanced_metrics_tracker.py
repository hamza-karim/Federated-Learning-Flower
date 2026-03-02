# Enhanced Metrics Tracker for Federated Learning
# Includes: Network metrics, Round completion breakdown, Client timing, Latency analysis

import time
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

class EnhancedMetricsTracker:
    """
    Comprehensive metrics tracker for FL with:
    - Network communication (bytes sent/received, throughput)
    - Round completion breakdown (client times, server idle, straggler effect)
    - Latency tracking (for future latency injection experiments)
    """

    def __init__(self):
        # ==========================================
        # NETWORK METRICS
        # ==========================================
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.communication_rounds = 0
        self.round_metrics = defaultdict(dict)

        # ==========================================
        # TIMING METRICS
        # ==========================================
        self.round_start_times = {}
        self.round_end_times = {}

        # NEW: Per-client timing tracking
        self.client_fit_start_times = defaultdict(dict)  # {round: {client_id: timestamp}}
        self.client_fit_end_times = defaultdict(dict)
        self.client_upload_start_times = defaultdict(dict)
        self.client_upload_end_times = defaultdict(dict)

        # NEW: Server-side timing
        self.aggregation_start_times = {}
        self.aggregation_end_times = {}
        self.server_idle_times = defaultdict(float)

        # NEW: Latency tracking (for future experiments)
        self.network_latency = defaultdict(dict)  # {round: {client_id: latency_ms}}

    # ==========================================
    # NETWORK METRICS CALCULATION
    # ==========================================

    def calculate_parameters_size(self, parameters) -> int:
        """
        Calculate size of model parameters in bytes.
        Supports: Flower Parameters, NumPy arrays, lists
        """
        total_size = 0

        if parameters is None:
            return 0

        # Handle Flower Parameters object
        if hasattr(parameters, "tensors"):
            try:
                for tensor in parameters.tensors:
                    if isinstance(tensor, (bytes, bytearray)):
                        total_size += len(tensor)
                    else:
                        total_size += sys.getsizeof(tensor)
            except Exception:
                total_size = sys.getsizeof(parameters)

        # Handle list of NumPy arrays
        elif isinstance(parameters, list):
            for param in parameters:
                if isinstance(param, np.ndarray):
                    total_size += int(param.nbytes)
                else:
                    total_size += sys.getsizeof(param)

        # Handle single NumPy array
        elif isinstance(parameters, np.ndarray):
            total_size = int(parameters.nbytes)

        else:
            total_size = sys.getsizeof(parameters)

        return int(total_size)

    def set_bytes_sent_outgoing(self, round_num: int, bytes_sent: int):
        """Track bytes sent from server to clients (model broadcast)"""
        self.round_metrics[round_num]['bytes_sent_outgoing'] = int(bytes_sent)

    def add_communication_overhead(
        self,
        round_num: int,
        bytes_sent: int,
        bytes_received: int,
        num_clients: int
    ):
        """
        Add communication metrics for a round.

        Args:
            round_num: FL round number
            bytes_sent: Bytes sent from server to clients
            bytes_received: Bytes received from clients to server
            num_clients: Number of clients that participated
        """
        # Use pre-calculated outgoing bytes if available
        if 'bytes_sent_outgoing' in self.round_metrics[round_num]:
            bytes_sent = int(self.round_metrics[round_num]['bytes_sent_outgoing'])
        else:
            bytes_sent = int(bytes_sent)

        self.round_metrics[round_num]['bytes_sent'] = int(bytes_sent)
        self.round_metrics[round_num]['bytes_received'] = int(bytes_received)
        self.round_metrics[round_num]['total_bytes'] = int(bytes_sent) + int(bytes_received)
        self.round_metrics[round_num]['num_clients_communicated'] = int(num_clients)

        # Update totals
        self.total_bytes_sent += int(bytes_sent)
        self.total_bytes_received += int(bytes_received)
        self.communication_rounds += 1

    # ==========================================
    # ROUND TIMING METHODS
    # ==========================================

    def start_round(self, round_num: int):
        """Mark the start of a FL round"""
        self.round_start_times[round_num] = time.time()

    def end_round(self, round_num: int):
        """Mark the end of a FL round"""
        self.round_end_times[round_num] = time.time()

    def get_round_duration(self, round_num: int) -> float:
        """Get total duration of a round in seconds"""
        if round_num in self.round_start_times and round_num in self.round_end_times:
            return self.round_end_times[round_num] - self.round_start_times[round_num]
        return 0.0

    # ==========================================
    # NEW: CLIENT-LEVEL TIMING TRACKING
    # ==========================================

    def record_client_fit_start(self, round_num: int, client_id: str):
        """Record when a client starts local training"""
        self.client_fit_start_times[round_num][client_id] = time.time()

    def record_client_fit_end(self, round_num: int, client_id: str):
        """Record when a client finishes local training"""
        self.client_fit_end_times[round_num][client_id] = time.time()

    def record_client_upload_start(self, round_num: int, client_id: str):
        """Record when client starts uploading results"""
        self.client_upload_start_times[round_num][client_id] = time.time()

    def record_client_upload_end(self, round_num: int, client_id: str):
        """Record when client finishes uploading results"""
        self.client_upload_end_times[round_num][client_id] = time.time()

    def get_client_training_time(self, round_num: int, client_id: str) -> float:
        """Get training time for a specific client in seconds"""
        if (round_num in self.client_fit_start_times and
            client_id in self.client_fit_start_times[round_num] and
            round_num in self.client_fit_end_times and
            client_id in self.client_fit_end_times[round_num]):
            return (self.client_fit_end_times[round_num][client_id] -
                   self.client_fit_start_times[round_num][client_id])
        return 0.0

    def get_client_upload_time(self, round_num: int, client_id: str) -> float:
        """Get upload time for a specific client in seconds"""
        if (round_num in self.client_upload_start_times and
            client_id in self.client_upload_start_times[round_num] and
            round_num in self.client_upload_end_times and
            client_id in self.client_upload_end_times[round_num]):
            return (self.client_upload_end_times[round_num][client_id] -
                   self.client_upload_start_times[round_num][client_id])
        return 0.0

    # ==========================================
    # NEW: SERVER-SIDE TIMING
    # ==========================================

    def record_aggregation_start(self, round_num: int):
        """Record when server starts aggregating client updates"""
        self.aggregation_start_times[round_num] = time.time()

    def record_aggregation_end(self, round_num: int):
        """Record when server finishes aggregation"""
        self.aggregation_end_times[round_num] = time.time()

    def calculate_server_idle_time(self, round_num: int):
        """
        Calculate time server spent waiting for clients.

        Server idle time = Time from first client finish to last client finish
        (i.e., waiting for stragglers)
        """
        if round_num not in self.client_fit_end_times:
            return 0.0

        client_end_times = list(self.client_fit_end_times[round_num].values())
        if len(client_end_times) < 2:
            return 0.0

        first_finish = min(client_end_times)
        last_finish = max(client_end_times)
        idle_time = last_finish - first_finish

        self.server_idle_times[round_num] = idle_time
        return idle_time

    # ==========================================
    # NEW: LATENCY TRACKING (for future experiments)
    # ==========================================

    def record_network_latency(self, round_num: int, client_id: str, latency_ms: float):
        """
        Record network latency for a client.

        Args:
            round_num: FL round number
            client_id: Client identifier
            latency_ms: Measured or injected latency in milliseconds
        """
        self.network_latency[round_num][client_id] = latency_ms

    # ==========================================
    # COMPUTE METRICS (Aggregations)
    # ==========================================

    def add_compute_metrics(
        self,
        round_num: int,
        aggregation_time: float,
        num_parameters: int,
        num_parameter_bytes: int
    ):
        """Add server-side computation metrics"""
        self.round_metrics[round_num]['aggregation_time'] = float(aggregation_time)
        self.round_metrics[round_num]['num_parameters'] = int(num_parameters)
        self.round_metrics[round_num]['num_parameter_bytes'] = int(num_parameter_bytes)

    # ==========================================
    # NEW: ROUND COMPLETION BREAKDOWN
    # ==========================================

    def calculate_round_breakdown(self, round_num: int) -> Dict:
        """
        Calculate comprehensive round completion metrics.

        Returns dict with:
        - Client training times (mean, std, min, max)
        - Client upload times (mean, std, min, max)
        - Server idle time
        - Aggregation delay
        - Straggler effect
        - Synchronization efficiency
        """
        breakdown = {}

        # ============================================
        # 1. CLIENT TRAINING TIMES
        # ============================================
        training_times = []
        for client_id in self.client_fit_start_times.get(round_num, {}).keys():
            t = self.get_client_training_time(round_num, client_id)
            if t > 0:
                training_times.append(t)

        if training_times:
            breakdown['client_training_mean'] = np.mean(training_times)
            breakdown['client_training_std'] = np.std(training_times)
            breakdown['client_training_min'] = np.min(training_times)
            breakdown['client_training_max'] = np.max(training_times)
            breakdown['client_training_median'] = np.median(training_times)

            # Straggler effect: difference between slowest and fastest
            breakdown['straggler_effect'] = np.max(training_times) - np.min(training_times)
        else:
            breakdown['client_training_mean'] = 0.0
            breakdown['client_training_std'] = 0.0
            breakdown['client_training_min'] = 0.0
            breakdown['client_training_max'] = 0.0
            breakdown['client_training_median'] = 0.0
            breakdown['straggler_effect'] = 0.0

        # ============================================
        # 2. CLIENT UPLOAD TIMES
        # ============================================
        upload_times = []
        for client_id in self.client_upload_start_times.get(round_num, {}).keys():
            t = self.get_client_upload_time(round_num, client_id)
            if t > 0:
                upload_times.append(t)

        if upload_times:
            breakdown['client_upload_mean'] = np.mean(upload_times)
            breakdown['client_upload_std'] = np.std(upload_times)
            breakdown['client_upload_min'] = np.min(upload_times)
            breakdown['client_upload_max'] = np.max(upload_times)
            breakdown['client_upload_median'] = np.median(upload_times)
        else:
            breakdown['client_upload_mean'] = 0.0
            breakdown['client_upload_std'] = 0.0
            breakdown['client_upload_min'] = 0.0
            breakdown['client_upload_max'] = 0.0
            breakdown['client_upload_median'] = 0.0

        # ============================================
        # 3. SERVER IDLE TIME
        # ============================================
        breakdown['server_idle_time'] = self.calculate_server_idle_time(round_num)

        # ============================================
        # 4. AGGREGATION TIME
        # ============================================
        if (round_num in self.aggregation_start_times and
            round_num in self.aggregation_end_times):
            breakdown['aggregation_time'] = (
                self.aggregation_end_times[round_num] -
                self.aggregation_start_times[round_num]
            )

            # Aggregation start delay: time from round start to aggregation start
            if round_num in self.round_start_times:
                breakdown['aggregation_start_delay'] = (
                    self.aggregation_start_times[round_num] -
                    self.round_start_times[round_num]
                )
        else:
            breakdown['aggregation_time'] = self.round_metrics[round_num].get('aggregation_time', 0.0)
            breakdown['aggregation_start_delay'] = 0.0

        # ============================================
        # 5. SYNCHRONIZATION EFFICIENCY
        # ============================================
        # Efficiency = Average training time / Max training time
        # Higher = better (less time wasted waiting)
        if training_times and breakdown['client_training_max'] > 0:
            breakdown['synchronization_efficiency'] = (
                breakdown['client_training_mean'] / breakdown['client_training_max']
            )
        else:
            breakdown['synchronization_efficiency'] = 1.0

        # ============================================
        # 6. NETWORK LATENCY (if available)
        # ============================================
        if round_num in self.network_latency:
            latencies = list(self.network_latency[round_num].values())
            breakdown['network_latency_mean'] = np.mean(latencies)
            breakdown['network_latency_std'] = np.std(latencies)
            breakdown['network_latency_min'] = np.min(latencies)
            breakdown['network_latency_max'] = np.max(latencies)

        return breakdown

    # ==========================================
    # ENHANCED PRINTING
    # ==========================================

    def print_round_summary(self, round_num: int):
        """Print comprehensive round summary with breakdown"""
        metrics = self.round_metrics.get(round_num, {})
        breakdown = self.calculate_round_breakdown(round_num)

        print(f"\n{'='*80}")
        print(f"ROUND {round_num} COMPREHENSIVE SUMMARY")
        print(f"{'='*80}")

        # ============================================
        # COMMUNICATION METRICS
        # ============================================
        print(f"\n📡 COMMUNICATION METRICS:")
        print(f"{'─'*80}")

        if 'bytes_sent' in metrics or 'bytes_received' in metrics:
            bytes_sent = metrics.get('bytes_sent', metrics.get('bytes_sent_outgoing', 0))
            bytes_received = metrics.get('bytes_received', 0)
            total_bytes = metrics.get('total_bytes', bytes_sent + bytes_received)
            num_clients = metrics.get('num_clients_communicated', 0)

            print(f"  Bytes Sent (Server→Clients):     {bytes_sent:>15,} bytes ({bytes_sent/1024/1024:>8.2f} MB)")
            print(f"  Bytes Received (Clients→Server): {bytes_received:>15,} bytes ({bytes_received/1024/1024:>8.2f} MB)")
            print(f"  Total Communication:             {total_bytes:>15,} bytes ({total_bytes/1024/1024:>8.2f} MB)")
            print(f"  Clients Participated:            {num_clients:>15}")

            # Calculate throughput
            round_duration = self.get_round_duration(round_num)
            if round_duration > 0:
                throughput_mbps = (total_bytes * 8) / (round_duration * 1_000_000)
                print(f"  Average Throughput:              {throughput_mbps:>15.2f} Mbps")

        # ============================================
        # TIMING BREAKDOWN
        # ============================================
        print(f"\n⏱️  TIMING BREAKDOWN:")
        print(f"{'─'*80}")

        print(f"  Round Duration:                  {self.get_round_duration(round_num):>15.2f} seconds")

        if 'client_training_mean' in breakdown:
            print(f"\n  Client Training Times:")
            print(f"    Mean:                          {breakdown['client_training_mean']:>15.2f} seconds")
            print(f"    Std Dev:                       {breakdown['client_training_std']:>15.2f} seconds")
            print(f"    Min (fastest):                 {breakdown['client_training_min']:>15.2f} seconds")
            print(f"    Max (slowest):                 {breakdown['client_training_max']:>15.2f} seconds")
            print(f"    Median:                        {breakdown['client_training_median']:>15.2f} seconds")

        if 'client_upload_mean' in breakdown and breakdown['client_upload_mean'] > 0:
            print(f"\n  Client Upload Times:")
            print(f"    Mean:                          {breakdown['client_upload_mean']:>15.2f} seconds")
            print(f"    Std Dev:                       {breakdown['client_upload_std']:>15.2f} seconds")
            print(f"    Min:                           {breakdown['client_upload_min']:>15.2f} seconds")
            print(f"    Max:                           {breakdown['client_upload_max']:>15.2f} seconds")

        print(f"\n  Server Metrics:")
        print(f"    Idle Time (waiting):           {breakdown.get('server_idle_time', 0.0):>15.2f} seconds")
        print(f"    Aggregation Time:              {breakdown.get('aggregation_time', 0.0):>15.4f} seconds")
        print(f"    Aggregation Start Delay:       {breakdown.get('aggregation_start_delay', 0.0):>15.2f} seconds")

        # ============================================
        # EFFICIENCY METRICS
        # ============================================
        print(f"\n📊 EFFICIENCY METRICS:")
        print(f"{'─'*80}")

        if 'straggler_effect' in breakdown:
            print(f"  Straggler Effect (max-min):      {breakdown['straggler_effect']:>15.2f} seconds")

        if 'synchronization_efficiency' in breakdown:
            eff_pct = breakdown['synchronization_efficiency'] * 100
            print(f"  Synchronization Efficiency:      {eff_pct:>15.1f}%")

            if eff_pct < 80:
                print(f"    ⚠️  WARNING: Low efficiency - significant straggler effect!")

        # ============================================
        # LATENCY (if available)
        # ============================================
        if 'network_latency_mean' in breakdown:
            print(f"\n🌐 NETWORK LATENCY:")
            print(f"{'─'*80}")
            print(f"  Mean Latency:                    {breakdown['network_latency_mean']:>15.2f} ms")
            print(f"  Std Dev:                         {breakdown['network_latency_std']:>15.2f} ms")
            print(f"  Min:                             {breakdown['network_latency_min']:>15.2f} ms")
            print(f"  Max:                             {breakdown['network_latency_max']:>15.2f} ms")

        # ============================================
        # COMPUTE METRICS
        # ============================================
        if 'num_parameters' in metrics:
            print(f"\n💻 COMPUTE METRICS:")
            print(f"{'─'*80}")
            print(f"  Model Parameters (elements):     {metrics.get('num_parameters', 0):>15,}")
            print(f"  Model Parameters (bytes):        {metrics.get('num_parameter_bytes', 0):>15,}")

        print(f"{'='*80}\n")

    def print_final_summary(self):
        """Print final summary with overall statistics"""
        print(f"\n{'='*80}")
        print(f"FEDERATED LEARNING FINAL SUMMARY")
        print(f"{'='*80}")

        # ============================================
        # OVERALL STATS
        # ============================================
        total_time = sum([self.get_round_duration(r) for r in self.round_start_times.keys()])

        print(f"\n📈 OVERALL STATISTICS:")
        print(f"{'─'*80}")
        print(f"  Total Communication Rounds:      {self.communication_rounds:>15}")
        print(f"  Total Training Time:             {total_time:>15.2f} seconds ({total_time/60:>8.2f} minutes)")

        # ============================================
        # COMMUNICATION SUMMARY
        # ============================================
        print(f"\n📡 TOTAL COMMUNICATION:")
        print(f"{'─'*80}")
        print(f"  Total Bytes Sent:                {self.total_bytes_sent:>15,} bytes ({self.total_bytes_sent/1024/1024:>8.2f} MB)")
        print(f"  Total Bytes Received:            {self.total_bytes_received:>15,} bytes ({self.total_bytes_received/1024/1024:>8.2f} MB)")
        total_comm = self.total_bytes_sent + self.total_bytes_received
        print(f"  Total Communication Overhead:    {total_comm:>15,} bytes ({total_comm/1024/1024:>8.2f} MB)")

        if total_time > 0:
            avg_throughput = (total_comm * 8) / (total_time * 1_000_000)
            print(f"  Average Overall Throughput:      {avg_throughput:>15.2f} Mbps")

        # ============================================
        # AGGREGATED TIMING METRICS
        # ============================================
        all_training_times = []
        all_upload_times = []
        all_idle_times = []
        all_straggler_effects = []
        all_sync_efficiencies = []

        for round_num in range(1, self.communication_rounds + 1):
            breakdown = self.calculate_round_breakdown(round_num)

            if 'client_training_mean' in breakdown and breakdown['client_training_mean'] > 0:
                all_training_times.append(breakdown['client_training_mean'])

            if 'client_upload_mean' in breakdown and breakdown['client_upload_mean'] > 0:
                all_upload_times.append(breakdown['client_upload_mean'])

            if 'server_idle_time' in breakdown:
                all_idle_times.append(breakdown['server_idle_time'])

            if 'straggler_effect' in breakdown:
                all_straggler_effects.append(breakdown['straggler_effect'])

            if 'synchronization_efficiency' in breakdown:
                all_sync_efficiencies.append(breakdown['synchronization_efficiency'])

        if all_training_times:
            print(f"\n⏱️  AVERAGE TIMING ACROSS ROUNDS:")
            print(f"{'─'*80}")
            print(f"  Avg Client Training Time:        {np.mean(all_training_times):>15.2f} seconds")
            print(f"  Avg Client Upload Time:          {np.mean(all_upload_times) if all_upload_times else 0:>15.2f} seconds")
            print(f"  Avg Server Idle Time:            {np.mean(all_idle_times):>15.2f} seconds")
            print(f"  Avg Straggler Effect:            {np.mean(all_straggler_effects):>15.2f} seconds")
            print(f"  Avg Sync Efficiency:             {np.mean(all_sync_efficiencies)*100:>15.1f}%")

        print(f"{'='*80}\n")

        # Save to file
        self.save_metrics_to_file()

    def save_metrics_to_file(self):
        """Save all metrics to JSON file with enhanced breakdown"""
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

        # Add per-round data with breakdown
        for round_num in sorted(self.round_metrics.keys()):
            breakdown = self.calculate_round_breakdown(round_num)

            output['per_round_metrics'][f'round_{round_num}'] = {
                'duration_seconds': self.get_round_duration(round_num),
                **self.round_metrics[round_num],
                'breakdown': breakdown
            }

        with open('fl_metrics.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("✅ Comprehensive metrics saved to fl_metrics.json")


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Example demonstrating the enhanced tracker
    tracker = EnhancedMetricsTracker()

    # Simulate Round 1
    tracker.start_round(1)

    # Simulate client training
    tracker.record_client_fit_start(1, "client_1")
    time.sleep(0.1)  # Simulate training
    tracker.record_client_fit_end(1, "client_1")

    tracker.record_client_fit_start(1, "client_2")
    time.sleep(0.15)  # Slower client (straggler)
    tracker.record_client_fit_end(1, "client_2")

    # Simulate aggregation
    tracker.record_aggregation_start(1)
    time.sleep(0.01)
    tracker.record_aggregation_end(1)

    # Add communication metrics
    tracker.add_communication_overhead(1, 1000000, 500000, 2)
    tracker.add_compute_metrics(1, 0.01, 100000, 400000)

    tracker.end_round(1)

    # Print summary
    tracker.print_round_summary(1)
    tracker.print_final_summary()