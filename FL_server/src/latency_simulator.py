# latency_simulator.py
# DER Network Latency Simulator
# Based on NREL Hybrid Communication Architecture Requirements

import time
import random
import numpy as np
from enum import Enum
from typing import Dict, List, Optional

class DERNetworkProfile(Enum):
    """
    DER communication network profiles based on real-world infrastructure.
    """
    GOOD_CONNECTIVITY = "good_connectivity"      # 70% of DERs
    POOR_CONNECTIVITY = "poor_connectivity"      # 30% of DERs
    # Baseline is not needed as a profile because disabled = 0 latency

class DERLatencyInjector:
    """
    Simulates heterogeneous DER network latency based on realistic infrastructure.
    
    When disabled: Injects 0.0s latency (Pure System Performance).
    """
    
    def __init__(
        self, 
        good_connectivity_percentage: float = 0.7,
        enable_latency: bool = True,
        seed: int = 42,
        verbose: bool = True
    ):
        self.good_percentage = good_connectivity_percentage
        self.enable_latency = enable_latency
        self.verbose = verbose
        
        # Client profile assignments
        self.client_profiles = {}  # {client_id: DERNetworkProfile}
        
        # Latency history for logging/analysis
        self.latency_history = {}  # {round_num: {client_id: latency_ms}}
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Define latency profiles (in milliseconds)
        # NREL / IEEE 2030.5 Specs
        self.latency_specs = {
            DERNetworkProfile.GOOD_CONNECTIVITY: {
                "mean_ms": 100, "std_ms": 25, "min_ms": 50, "max_ms": 150,
                "infrastructure": "Fiber, Cable, 4G LTE"
            },
            DERNetworkProfile.POOR_CONNECTIVITY: {
                "mean_ms": 250, "std_ms": 25, "min_ms": 200, "max_ms": 300,
                "infrastructure": "DSL, 3G, Narrowband PLC"
            }
        }
        
        if self.verbose:
            self._print_initialization_summary()
    
    def _print_initialization_summary(self):
        """Print initialization configuration"""
        print(f"\n{'='*70}")
        print(f"DER Network Latency Simulator Initialized")
        print(f"{'='*70}")
        
        if not self.enable_latency:
            print(f"Status: DISABLED")
            print(f"  Action: No latency will be injected.")
            print(f"  Goal: Measure pure hardware/LAN performance.")
        else:
            print(f"Status: ENABLED")
            print(f"  Good Connectivity ({self.good_percentage*100:.0f}%): 100ms ± 25ms")
            print(f"  Poor Connectivity ({(1-self.good_percentage)*100:.0f}%): 250ms ± 25ms")
        
        print(f"{'='*70}\n")
    
    def assign_client_profile(
        self, 
        client_id: int, 
        total_clients: int
    ) -> DERNetworkProfile:
        """Assign network profile to a client (Only used if latency enabled)."""
        # Check if already assigned
        if client_id in self.client_profiles:
            return self.client_profiles[client_id]
        
        # Deterministic assignment
        num_good_clients = int(total_clients * self.good_percentage)
        
        if client_id <= num_good_clients:
            profile = DERNetworkProfile.GOOD_CONNECTIVITY
        else:
            profile = DERNetworkProfile.POOR_CONNECTIVITY
        
        self.client_profiles[client_id] = profile
        
        # Verbose logging
        if self.verbose and self.enable_latency:
            spec = self.latency_specs[profile]
            print(f"[DER Network] Client {client_id:2d} → {profile.value.upper()}")
        
        return profile
    
    def inject_latency(
        self, 
        client_id: int, 
        total_clients: int, 
        round_num: int = 0
    ) -> float:
        """
        Inject network latency delay.
        
        STRICT BEHAVIOR:
        - If enable_latency is False: Returns 0.0 immediately.
        - If enable_latency is True: Sleeps for calculated ms.
        """
        # ============================================
        # 1. STRICT CHECK: IF DISABLED, DO NOTHING
        # ============================================
        if not self.enable_latency:
            return 0.0

        # ============================================
        # 2. CALCULATE AND INJECT
        # ============================================
        profile = self.assign_client_profile(client_id, total_clients)
        spec = self.latency_specs[profile]
        
        # Sample latency
        latency = np.random.normal(spec['mean_ms'], spec['std_ms'])
        latency = np.clip(latency, spec['min_ms'], spec['max_ms'])
        
        # Log it
        if round_num not in self.latency_history:
            self.latency_history[round_num] = {}
        self.latency_history[round_num][client_id] = latency
        
        # Sleep (Inject Delay)
        time.sleep(latency / 1000.0) 
        
        if self.verbose:
            print(f"[DER Network] Client {client_id:2d}: Injected {latency:.1f}ms latency")
        
        return float(latency)
    
    def print_round_summary(self, round_num: int):
        """Print latency summary (Only if enabled)"""
        if not self.enable_latency:
            return # Print nothing if disabled
            
        if round_num not in self.latency_history:
            print(f"\n[DER Network] No latency data for Round {round_num}")
            return
        
        latencies = list(self.latency_history[round_num].values())
        print(f"\n[DER Network] Round {round_num} Summary: Mean Latency = {np.mean(latencies):.2f} ms")

    def export_statistics(self) -> Dict:
        """Export statistics (Empty if disabled)"""
        if not self.enable_latency:
            return {"enabled": False, "message": "Latency simulation was disabled."}
            
        return {
            "enabled": True,
            "round_statistics": self.latency_history
        }