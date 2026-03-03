import os
import io
import re
import json
import base64
import numpy as np
import subprocess
import pandas as pd
import streamlit as st
import PIL.Image as Image
from graphviz import Digraph
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

st.set_page_config(page_title="C2SR - FL Deployment", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 1. CSS STYLING
# ==========================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #e0e8f0;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3c72;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2a5298;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .device-status {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #4caf50; }
    .status-offline { background-color: #f44336; }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .device-badge {
        font-size: 0.75em; 
        padding: 2px 8px; 
        border-radius: 12px; 
        margin-left: 8px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-edge { background-color: #e0f7fa; color: #006064; border: 1px solid #b2ebf2; }
    .badge-lambda { background-color: #f3e5f5; color: #4a148c; border: 1px solid #e1bee7; }
    .badge-nano   { background-color: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
    .badge-global { background-color: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
    .arch-card {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border: 2px solid transparent;
        cursor: pointer;
        transition: all 0.2s;
        margin-bottom: 0.5rem;
    }
    .arch-selected-cfl {
        border-color: #2a5298;
        background: linear-gradient(135deg, #e8f0fe, #c7d9f9);
    }
    .arch-selected-hfl {
        border-color: #006400;
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    }
    .arch-unselected {
        border-color: #e0e0e0;
        background: #f8f9fa;
    }
    .tier-box {
        background: white;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 5px solid;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .tier-global { border-color: #1b5e20; }
    .tier-edge   { border-color: #e65100; }
    .tier-client { border-color: #4a148c; }
    .info-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #e3f2fd;
        color: #1565c0;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEADER & LOGO
# ==========================================
LOGO_PATH = "./Picture1.png" 

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #006400 0%, #00a86b 100%);
                padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0;">
            Federated Learning Deployment Platform
        </h1>
        <p style="color: #d4f8e8; font-size: 1rem; margin-top: 0.3rem; margin-bottom: 0;">
            Federated Learning Container Management in the C2SR Edge Testbed
        </p>
    </div>
    """, unsafe_allow_html=True)

try:
    with open(LOGO_PATH, "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
except:
    encoded_logo = ""

with col2:
    if encoded_logo:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #006400 0%, #00a86b 100%);
                    padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;">
            <img src="data:image/png;base64,{encoded_logo}" 
                 alt="C2SR Logo" style="width: 100%; object-fit: contain; max-height: 110px;">
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 3. DEVICE REGISTRY
# ==========================================
AVAILABLE_DEVICES = {
    "10.226.44.86": {
        "hostname": "c2sragx04",
        "display_name": "AGX 04",
        "type": "EDGE"
    },
    "10.226.47.0": {
        "hostname": "c2srnano07",
        "display_name": "Nano 07",
        "type": "EDGE"
    },
    "10.226.47.108": {
        "hostname": "c2srnano08",
        "display_name": "Nano 08",
        "type": "EDGE"
    },
    "10.226.46.8": {
        "hostname": "hamzakarim",
        "display_name": "Nano 10",
        "type": "EDGE"
    },
    "10.226.47.64": {
        "hostname": "hamzakarim",
        "display_name": "Nano 13",
        "type": "EDGE"
    },
    "10.226.31.254": { 
        "hostname": "hamza.karim",
        "display_name": "Lambda Server",
        "type": "LAMBDA"
    }
}

# Only Jetson edge devices (no Lambda) for server/aggregator roles
EDGE_ONLY_DEVICES = {k: v for k, v in AVAILABLE_DEVICES.items() if v["type"] == "EDGE"}

def check_device_status(hostname, ip):
    try:
        if ip == "127.0.0.1" or ip == "localhost":
            return True
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             f"{hostname}@{ip}", "echo 'connected'"],
            capture_output=True, timeout=3
        )
        return result.returncode == 0
    except:
        return False

# ==========================================
# 4. SYSTEM OVERVIEW METRICS
# ==========================================
st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Available Devices", len(AVAILABLE_DEVICES))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    configured = len(st.session_state.get('clients', []))
    st.metric("Configured Devices", configured)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    total_containers = 0
    for d in st.session_state.get('clients', []):
        if 'client_ids' in d:
            total_containers += len(d['client_ids'])
        else:
            total_containers += d['end'] - d['start'] + 1
    st.metric("Total Containers", total_containers)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. DEVICE STATUS CHECK
# ==========================================
st.markdown('<div class="section-header">Device Status</div>', unsafe_allow_html=True)

if 'device_status_cache' not in st.session_state:
    st.session_state.device_status_cache = {}

with st.expander("View All Available Devices", expanded=False):
    if st.button("🔄 Refresh Status", key="refresh_status"):
        st.session_state.device_status_cache = {}
        for ip, device_info in AVAILABLE_DEVICES.items():
            st.session_state.device_status_cache[ip] = check_device_status(device_info["hostname"], ip)
        st.rerun()
    
    if not st.session_state.device_status_cache:
        with st.spinner("Checking device status..."):
            for ip, device_info in AVAILABLE_DEVICES.items():
                st.session_state.device_status_cache[ip] = check_device_status(device_info["hostname"], ip)
    
    cols = st.columns(3)
    for idx, (ip, device_info) in enumerate(AVAILABLE_DEVICES.items()):
        with cols[idx % 3]:
            is_online = st.session_state.device_status_cache.get(ip, False)
            status_class = "status-online" if is_online else "status-offline"
            status_text = "Online" if is_online else "Offline"
            badge_class = "badge-lambda" if device_info['type'] == "LAMBDA" else "badge-edge"
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #2a5298; margin-bottom: 0.8rem;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>{device_info["display_name"]}</strong> 
                        <span class="device-badge {badge_class}">{device_info['type']}</span><br>
                        <small style="color: #666;">{ip}</small>
                    </div>
                    <div>
                        <span class="device-status {status_class}"></span>
                        <small>{status_text}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 6. ARCHITECTURE SELECTION  ← NEW
# ==========================================
st.markdown('<div class="section-header">FL Architecture</div>', unsafe_allow_html=True)

if 'fl_architecture' not in st.session_state:
    st.session_state.fl_architecture = "Centralized FL"

arch_col1, arch_col2 = st.columns(2)

with arch_col1:
    cfl_selected = st.session_state.fl_architecture == "Centralized FL"
    cfl_style = "arch-selected-cfl" if cfl_selected else "arch-unselected"
    st.markdown(f"""
    <div class="arch-card {cfl_style}">
        <h3 style="margin:0; color: #1e3c72;">🖥️ Centralized FL</h3>
        <p style="margin: 0.5rem 0 0 0; color: #444; font-size: 0.9rem;">
            All clients communicate directly with a single FL server.<br>
            <strong>Server → N Clients</strong> (2-tier)
        </p>
        <span class="info-pill">Current Setup</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Centralized FL", use_container_width=True, key="sel_cfl",
                 type="primary" if cfl_selected else "secondary"):
        st.session_state.fl_architecture = "Centralized FL"
        st.rerun()

with arch_col2:
    hfl_selected = st.session_state.fl_architecture == "HFL"
    hfl_style = "arch-selected-hfl" if hfl_selected else "arch-unselected"
    st.markdown(f"""
    <div class="arch-card {hfl_style}">
        <h3 style="margin:0; color: #006400;">🌿 Hierarchical FL (HFL)</h3>
        <p style="margin: 0.5rem 0 0 0; color: #444; font-size: 0.9rem;">
            Clients → Nano Edge Aggregators → AGX Global Server.<br>
            <strong>3-Tier: Client → Edge → Global</strong>
        </p>
        <span class="info-pill" style="background:#e8f5e9; color:#1b5e20;">MQW-HierFAVG</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select HFL", use_container_width=True, key="sel_hfl",
                 type="primary" if hfl_selected else "secondary"):
        st.session_state.fl_architecture = "HFL"
        st.rerun()

st.markdown("---")

# ==========================================
# 7. FL CONFIGURATION  (Architecture-aware)
# ==========================================
st.markdown('<div class="section-header">FL Configuration</div>', unsafe_allow_html=True)

# ── Shared hyperparameters (always visible) ──────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    model = st.selectbox("Model", ["lstm", "bilstm"])
    algo  = st.selectbox("Algorithm", ["fedavg", "fedprox"])
with col2:
    total_clients = st.number_input("Total Clients", min_value=1, value=20)
    epochs        = st.number_input("Epochs/Client", min_value=1, max_value=50, value=5)
with col3:
    threshold_percentile = st.number_input("Anomaly Threshold (%)", min_value=90.0, max_value=100.0,
                                           value=99.0, step=0.1)
with col4:
    if st.session_state.fl_architecture == "HFL":
        local_rounds  = st.number_input("Local Rounds (τ₁)", min_value=1, max_value=20, value=3,
                                         help="Rounds each Nano runs with its clients before uploading to AGX")
        global_rounds = st.number_input("Global Rounds (τ₂)", min_value=1, max_value=50, value=10,
                                         help="Rounds AGX runs across all Nano aggregators")
        use_mqw = st.checkbox("Quality Weighting (MQW)", value=True,
                              help="Enable MAE-based quality weighting at edge tier")
    else:
        # Centralized FL: single server config in col4 area
        pass

# ── Architecture-specific server config ──────────────────────────────────────
if st.session_state.fl_architecture == "Centralized FL":
    # ── Original 2-tier setup ────────────────────────────────────────────────
    st.markdown("#### 🖥️ Server Configuration")
    c1, c2 = st.columns(2)
    with c1:
        server_device = st.selectbox(
            "Server Device",
            options=list(AVAILABLE_DEVICES.keys()),
            format_func=lambda x: f"{AVAILABLE_DEVICES[x]['display_name']} ({x})"
        )
        server_ip   = server_device
        server_port = st.text_input("Port", "8080")
    with c2:
        st.markdown("""
        <div style="background:#e8f0fe; padding:1rem; border-radius:8px; margin-top:1.8rem;">
            <strong>2-Tier Architecture</strong><br>
            <small>Server receives uploads from all N clients directly.<br>
            Network overhead scales with N.</small>
        </div>
        """, unsafe_allow_html=True)

    # Store for downstream use
    hfl_clusters = []

else:
    # ── 3-Tier HFL setup ────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(to right,#e8f5e9,#f1f8e9);
                padding: 0.8rem 1.2rem; border-radius: 8px; margin-bottom: 1.2rem;
                border-left: 4px solid #2e7d32;">
        <strong style="color:#1b5e20;">HFL Mode Active</strong> — Configure all three tiers below.
        Clients will connect to their assigned Nano, not the AGX directly.
    </div>
    """, unsafe_allow_html=True)

    # ── Tier 3: Global Aggregator (AGX) ─────────────────────────────────────
    st.markdown("""
    <div class="tier-box tier-global">
        <strong style="color:#1b5e20; font-size:1.05rem;">🌐 Tier 3 — Global Aggregator (AGX)</strong>
        <p style="margin:0.3rem 0 0 0; color:#555; font-size:0.88rem;">
            Aggregates models from all Nano edge servers. Runs <code>global_server.py</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        agx_device = st.selectbox(
            "Global Aggregator Device",
            options=list(EDGE_ONLY_DEVICES.keys()),
            format_func=lambda x: f"{AVAILABLE_DEVICES[x]['display_name']} ({x})",
            index=0,
            key="agx_device"
        )
        server_ip   = agx_device
        server_port = st.text_input("AGX Port", "8080", key="agx_port")
    with g2:
        st.markdown(f"""
        <div style="background:#e8f5e9; padding:0.9rem; border-radius:8px; margin-top:1.8rem;">
            <strong>Selected:</strong> {AVAILABLE_DEVICES[agx_device]['display_name']}<br>
            <small>IP: {agx_device} | Port: {server_port}</small><br>
            <small>Expects <strong>1 upload per Nano</strong> per global round (N-independent)</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tier 2: Nano Edge Aggregators ────────────────────────────────────────
    st.markdown("""
    <div class="tier-box tier-edge">
        <strong style="color:#e65100; font-size:1.05rem;">📡 Tier 2 — Edge Aggregators (Nanos)</strong>
        <p style="margin:0.3rem 0 0 0; color:#555; font-size:0.88rem;">
            Each Nano is a cluster head. Runs <code>edge_aggregator.py</code> (server to clients on port 8081,
            client to AGX on port 8080).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize HFL clusters in session state
    if 'hfl_clusters' not in st.session_state:
        st.session_state.hfl_clusters = [
            {"nano_ip": "10.226.47.0",   "nano_port": "8081", "client_ids": "1-5"},
            {"nano_ip": "10.226.47.108", "nano_port": "8081", "client_ids": "6-10"},
            {"nano_ip": "10.226.46.8",   "nano_port": "8081", "client_ids": "11-15"},
            {"nano_ip": "10.226.47.64",  "nano_port": "8081", "client_ids": "16-20"},
        ]

    # Filter available nano options (exclude whatever is chosen as AGX)
    nano_options = [ip for ip in EDGE_ONLY_DEVICES.keys() if ip != agx_device]

    clusters_to_remove = []
    for cidx, cluster in enumerate(st.session_state.hfl_clusters):
        with st.container():
            st.markdown(f"""
            <div style="background: #fff8f0; padding: 0.6rem 1rem; border-radius: 6px;
                        border: 1px solid #ffcc80; margin-bottom: 0.4rem;">
                <strong style="color:#e65100;">Cluster {cidx + 1}</strong>
            </div>
            """, unsafe_allow_html=True)

            cc1, cc2, cc3, cc4 = st.columns([3, 1, 3, 1])
            with cc1:
                default_idx = nano_options.index(cluster["nano_ip"]) if cluster["nano_ip"] in nano_options else 0
                selected_nano = st.selectbox(
                    "Nano Device",
                    options=nano_options,
                    format_func=lambda x: f"{AVAILABLE_DEVICES[x]['display_name']} ({x})",
                    index=default_idx,
                    key=f"nano_dev_{cidx}"
                )
                st.session_state.hfl_clusters[cidx]["nano_ip"] = selected_nano

            with cc2:
                port_val = st.text_input("Port", value=cluster["nano_port"], key=f"nano_port_{cidx}")
                st.session_state.hfl_clusters[cidx]["nano_port"] = port_val

            with cc3:
                client_ids_raw = st.text_input(
                    "Client IDs (e.g. 1-5, 7, 9)",
                    value=cluster["client_ids"],
                    key=f"nano_clients_{cidx}",
                    help="Ranges like 1-5, singles like 7, or comma-separated"
                )
                st.session_state.hfl_clusters[cidx]["client_ids"] = client_ids_raw

                # Preview parsed IDs
                try:
                    parsed = []
                    for part in [p.strip() for p in client_ids_raw.split(',')]:
                        if '-' in part:
                            s, e = map(int, part.split('-'))
                            parsed.extend(range(s, e + 1))
                        else:
                            parsed.append(int(part))
                    parsed = sorted(set(parsed))
                    st.caption(f"→ {len(parsed)} clients: {parsed}")
                except:
                    st.caption("⚠️ Invalid format")

            with cc4:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"rm_cluster_{cidx}", help="Remove this cluster"):
                    clusters_to_remove.append(cidx)

    for idx in sorted(clusters_to_remove, reverse=True):
        st.session_state.hfl_clusters.pop(idx)
    if clusters_to_remove:
        st.rerun()

    add_col, _ = st.columns([1, 3])
    with add_col:
        if st.button("➕ Add Cluster", use_container_width=True):
            used_nanos = [c["nano_ip"] for c in st.session_state.hfl_clusters]
            available_nanos = [ip for ip in nano_options if ip not in used_nanos]
            new_nano = available_nanos[0] if available_nanos else nano_options[0]
            next_start = max(
                [max([int(p) for p in c["client_ids"].replace('-', ',').split(',') if p.strip().isdigit()] or [0])
                 for c in st.session_state.hfl_clusters], default=0
            ) + 1
            st.session_state.hfl_clusters.append({
                "nano_ip": new_nano,
                "nano_port": "8081",
                "client_ids": f"{next_start}-{next_start + 4}"
            })
            st.rerun()

    # Build hfl_clusters parsed for use downstream
    hfl_clusters = []
    for cidx, cluster in enumerate(st.session_state.hfl_clusters):
        try:
            parsed_ids = []
            for part in [p.strip() for p in cluster["client_ids"].split(',')]:
                if '-' in part:
                    s, e = map(int, part.split('-'))
                    parsed_ids.extend(range(s, e + 1))
                elif part.isdigit():
                    parsed_ids.append(int(part))
            hfl_clusters.append({
                "cluster_idx": cidx + 1,
                "nano_ip": cluster["nano_ip"],
                "nano_port": cluster["nano_port"],
                "nano_hostname": AVAILABLE_DEVICES[cluster["nano_ip"]]["hostname"],
                "nano_display": AVAILABLE_DEVICES[cluster["nano_ip"]]["display_name"],
                "client_ids": sorted(set(parsed_ids))
            })
        except:
            pass

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tier 1: FL Clients ────────────────────────────────────────────────────
    st.markdown("""
    <div class="tier-box tier-client">
        <strong style="color:#4a148c; font-size:1.05rem;">💻 Tier 1 — FL Clients (Lambda)</strong>
        <p style="margin:0.3rem 0 0 0; color:#555; font-size:0.88rem;">
            Clients are configured in the <strong>Deployment Targets</strong> section below.
            In HFL mode, each client's <code>SERVER_IP</code> is auto-set to its assigned Nano,
            not the AGX.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show assignment summary
    if hfl_clusters:
        summary_cols = st.columns(len(hfl_clusters))
        for cidx, cluster in enumerate(hfl_clusters):
            with summary_cols[cidx]:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 8px;
                            border-top: 3px solid #e65100; text-align:center;">
                    <strong>{cluster['nano_display']}</strong><br>
                    <small style="color:#666;">Cluster {cluster['cluster_idx']}</small><br>
                    <span style="color:#4a148c; font-weight:600;">{len(cluster['client_ids'])} clients</span><br>
                    <small>IDs: {cluster['client_ids'][:3]}{'...' if len(cluster['client_ids']) > 3 else ''}</small>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# 8. DOCKER IMAGE CONFIGURATION
# ==========================================
st.markdown("### 🐳 Docker Images")
st.caption("Specify the exact Docker image names to use for each architecture.")

col_img1, col_img2 = st.columns(2)
with col_img1:
    image_edge = st.text_input(
        "Edge Device Image (ARM64)", 
        "hamzakarim07/flwr_client_hfl:latest", 
        help="Used for Jetson Nano and AGX devices"
    )
with col_img2:
    image_lambda = st.text_input(
        "Lambda Server Image (x86)", 
        "hamzakarim07/flwr_client_lambda:latest", 
        help="Used for the Lambda Server containers"
    )

if st.session_state.fl_architecture == "HFL":
    col_img3, col_img4 = st.columns(2)
    with col_img3:
        image_global_server = st.text_input(
            "Global Server Image (ARM64)",
            "hamzakarim07/flwr_global_server:latest",
            help="Runs global_server.py on AGX"
        )
    with col_img4:
        image_edge_aggregator = st.text_input(
            "Edge Aggregator Image (ARM64)",
            "hamzakarim07/flwr_edge_aggregator:latest",
            help="Runs edge_aggregator.py on each Nano"
        )

st.markdown("---")

# ==========================================
# 9. DEPLOYMENT TARGETS  (unchanged section)
# ==========================================
st.markdown('<div class="section-header">Deployment Targets</div>', unsafe_allow_html=True)

if st.session_state.fl_architecture == "HFL" and hfl_clusters:
    st.info(
        "ℹ️ **HFL Mode**: Add your Lambda Server below. Clients will automatically get "
        "`SERVER_IP` set to their assigned Nano based on the cluster configuration above."
    )

if 'clients' not in st.session_state:
    st.session_state.clients = []

with st.expander("➕ Add New Target Device", expanded=len(st.session_state.clients) == 0):
    with st.form("add_device", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            selected_ip = st.selectbox(
                "Select Device",
                options=list(AVAILABLE_DEVICES.keys()),
                format_func=lambda x: f"{AVAILABLE_DEVICES[x]['display_name']} [{AVAILABLE_DEVICES[x]['type']}]"
            )
        with c2:
            client_range = st.text_input(
                "Client IDs (e.g., 1-10, 15)", 
                value="1",
                help="Enter ranges (1-3), single IDs (5), or comma-separated (1,3,5)"
            )

        add_btn = st.form_submit_button("Add Device to Deployment", use_container_width=True)

        if add_btn:
            try:
                client_ids = []
                parts = [p.strip() for p in client_range.split(',')]
                for part in parts:
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        client_ids.extend(range(start, end + 1))
                    else:
                        client_ids.append(int(part))
                client_ids = sorted(set(client_ids))
                if not client_ids:
                    st.error("⚠️ No valid client IDs entered")
                else:
                    dev_info = AVAILABLE_DEVICES[selected_ip]
                    st.session_state.clients.append({
                        "hostname": dev_info["hostname"],
                        "display_name": dev_info["display_name"],
                        "ip": selected_ip,
                        "type": dev_info["type"], 
                        "client_ids": client_ids
                    })
                    st.rerun()
            except ValueError:
                st.error("⚠️ Please enter valid format (e.g., 1-3, 5, 7 or 1,5,7)")

if st.session_state.clients:
    st.markdown(f"""
    <div style="background: linear-gradient(to right, #e8f4f8, #f0f9ff); 
                padding: 0.8rem 1.2rem; border-radius: 6px; margin-bottom: 1rem;
                border-left: 4px solid #2a5298;">
        <strong style="color: #1e3c72;">
            {len(st.session_state.clients)} device(s) configured for deployment
        </strong>
    </div>
    """, unsafe_allow_html=True)
    
    for idx, device in enumerate(st.session_state.clients):
        badge_cls = "badge-lambda" if device['type'] == "LAMBDA" else "badge-edge"
        
        # In HFL mode: show which Nano each client maps to
        hfl_routing_info = ""
        if st.session_state.fl_architecture == "HFL" and hfl_clusters:
            routing_lines = []
            for cid in device['client_ids']:
                assigned = None
                for cluster in hfl_clusters:
                    if cid in cluster['client_ids']:
                        assigned = cluster
                        break
                if assigned:
                    routing_lines.append(f"Client {cid} → {assigned['nano_display']}")
                else:
                    routing_lines.append(f"Client {cid} → ⚠️ Unassigned")
            hfl_routing_info = " | ".join(routing_lines[:3])
            if len(routing_lines) > 3:
                hfl_routing_info += f" + {len(routing_lines)-3} more"

        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; 
                    border: 1px solid #e0e0e0; margin-bottom: 1rem; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**{device['display_name']}** <span class='device-badge {badge_cls}'>{device['type']}</span>", unsafe_allow_html=True)
            st.caption(f"IP: {device['ip']}")
            if hfl_routing_info:
                st.caption(f"🔀 {hfl_routing_info}")
        with col2:
            ids_str = ', '.join(map(str, device['client_ids']))
            st.write(f"Clients: {ids_str}")
            st.caption(f"({len(device['client_ids'])} containers)")
        with col3:
            if st.button("Remove", key=f"rm_{idx}", use_container_width=True):
                st.session_state.clients.pop(idx)
                st.rerun()
            if st.button("Cleanup", key=f"clean_{idx}", use_container_width=True):
                with st.spinner(f"Cleaning {device['display_name']}..."):
                    cleanup_cmd = "docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f"
                    try:
                        if device['ip'] == "127.0.0.1" or device['ip'] == "localhost":
                            subprocess.run(cleanup_cmd, shell=True, check=True, executable='/bin/bash')
                        else:
                            subprocess.run(
                                ["ssh", "-o", "StrictHostKeyChecking=no",
                                 f"{device['hostname']}@{device['ip']}", cleanup_cmd],
                                check=True, capture_output=True
                            )
                        st.success(f"✓ Cleaned {device['display_name']}")
                    except subprocess.CalledProcessError:
                        st.error(f"✗ Failed to cleanup {device['display_name']}")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No devices configured for deployment yet. Add your first device above.")

# ==========================================
# 10. TOPOLOGY VISUALIZATION  (Architecture-aware)
# ==========================================
st.markdown('<div class="section-header">Network Topology</div>', unsafe_allow_html=True)

dot = Digraph(format="png")
dot.attr(rankdir='TB', bgcolor='transparent', fontname='Helvetica', fontsize='10')
dot.attr('edge', penwidth='2')

if st.session_state.fl_architecture == "Centralized FL":
    # ── 2-tier: original topology ────────────────────────────────────────────
    server_display = AVAILABLE_DEVICES[server_ip]["display_name"]
    dot.node(
        'server',
        f"🖥️ {server_display}\n{server_ip}:{server_port}\nFL Server\n{algo.upper()}",
        shape='box', style='filled,rounded,bold',
        fillcolor='#4caf50', fontcolor='white', penwidth='2'
    )
    for idx, device in enumerate(st.session_state.clients):
        with dot.subgraph(name=f'cluster_{idx}') as c:
            c.attr(style='rounded,dashed', color='#2a5298', label=f"{device['display_name']}")
            c.attr(rank='same')
            for cid in device['client_ids']:
                client_node = f"{device['hostname']}_c{cid}"
                is_online = st.session_state.device_status_cache.get(device['ip'], True)
                fill_color = '#a8e6a1' if is_online else '#fca5a5'
                status_emoji = '🟢' if is_online else '🔴'
                c.node(client_node, f"{status_emoji} Client {cid}",
                       shape='circle', style='filled,rounded',
                       fillcolor=fill_color, fontcolor='black')
                edge_style = 'solid' if is_online else 'dashed'
                edge_color = '#2a5298' if is_online else '#ff6b6b'
                dot.edge('server', client_node, style=edge_style, color=edge_color)

else:
    # ── 3-tier: HFL topology ─────────────────────────────────────────────────
    agx_display = AVAILABLE_DEVICES[server_ip]["display_name"]
    dot.node(
        'agx',
        f"🌐 {agx_display}\n{server_ip}:{server_port}\nGlobal Aggregator\nglobal_server.py",
        shape='box', style='filled,rounded,bold',
        fillcolor='#1b5e20', fontcolor='white', penwidth='3'
    )

    for cluster in hfl_clusters:
        nano_node = f"nano_{cluster['cluster_idx']}"
        nano_display = cluster['nano_display']
        is_nano_online = st.session_state.device_status_cache.get(cluster['nano_ip'], True)
        nano_fill = '#ff8f00' if is_nano_online else '#fca5a5'
        dot.node(
            nano_node,
            f"📡 {nano_display}\n{cluster['nano_ip']}:{cluster['nano_port']}\nEdge Aggregator\nedge_aggregator.py",
            shape='box', style='filled,rounded',
            fillcolor=nano_fill, fontcolor='white', penwidth='2'
        )
        dot.edge('agx', nano_node, color='#1b5e20', penwidth='2',
                 label=f"Backhaul")

        # Add client nodes under each Nano
        with dot.subgraph(name=f'cluster_nano_{cluster["cluster_idx"]}') as c:
            c.attr(style='rounded,dashed', color='#e65100',
                   label=f"Cluster {cluster['cluster_idx']}")
            for cid in cluster['client_ids']:
                client_node = f"client_{cid}"
                # Check if any deployment device has this client
                is_online = True
                for dev in st.session_state.clients:
                    if cid in dev['client_ids']:
                        is_online = st.session_state.device_status_cache.get(dev['ip'], True)
                        break
                fill_color = '#a8e6a1' if is_online else '#fca5a5'
                status_emoji = '🟢' if is_online else '🔴'
                c.node(client_node, f"{status_emoji} Client {cid}",
                       shape='circle', style='filled,rounded',
                       fillcolor=fill_color, fontcolor='black')
                dot.edge(nano_node, client_node, color='#4a148c',
                         style='solid' if is_online else 'dashed')

st.graphviz_chart(dot)
st.markdown("---")

# ==========================================
# 11. DEPLOYMENT CONTROL  (Architecture-aware)
# ==========================================
st.markdown('<div class="section-header">Deployment Control</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.session_state.fl_architecture == "Centralized FL":
        # ── Original Centralized FL deploy ───────────────────────────────────
        if st.session_state.clients:
            if st.button("🚀 Deploy to All Devices", type="primary", use_container_width=True):
                for device in st.session_state.clients:
                    with st.expander(f"Deploying to {device['display_name']}...", expanded=True):
                        if device['type'] == "LAMBDA":
                            target_image = image_lambda
                            docker_flags = "--privileged --gpus all"
                        else:
                            target_image = image_edge
                            docker_flags = "--runtime=nvidia --gpus all"

                        st.write(f"🔹 Target Image: `{target_image}`")
                        st.write(f"⚙️ Runtime Flags: `{docker_flags}`")

                        script = f"""
                        docker pull {target_image}
                        docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f
                        """
                        for cid in device['client_ids']:
                            script += f"""
docker run -d --name flwr-client{cid} \\
  {docker_flags} \\
  --net=host \\
  --cap-add=NET_ADMIN \\
  -e CLIENT_ID={cid} \\
  -e TOTAL_CLIENTS={total_clients} \\
  -e EPOCHS={epochs} \\
  -e MODEL={model} \\
  -e ALGO={algo} \\
  -e SERVER_IP={server_ip} \\
  -e SERVER_PORT={server_port} \\
  -e THRESHOLD_PERCENTILE={threshold_percentile} \\
  {target_image}
"""
                        try:
                            if device['ip'] in ("127.0.0.1", "localhost"):
                                subprocess.run(script, shell=True, check=True, executable='/bin/bash')
                            else:
                                subprocess.run(
                                    ["ssh", "-o", "StrictHostKeyChecking=no",
                                     f"{device['hostname']}@{device['ip']}", script],
                                    check=True, capture_output=True, text=True
                                )
                            st.success(f"✓ Deployed {len(device['client_ids'])} containers to {device['display_name']}")
                        except subprocess.CalledProcessError as e:
                            st.error(f"✗ Deployment failed on {device['display_name']}")
                            with st.expander("Error details"):
                                st.code(e.stderr if hasattr(e, 'stderr') else str(e))
        else:
            st.warning("⚠️ Add at least one device to begin deployment")

    else:
        # ── HFL Deploy ───────────────────────────────────────────────────────
        if st.button("🚀 Deploy HFL (All Tiers)", type="primary", use_container_width=True):
            if not hfl_clusters:
                st.error("⚠️ No clusters configured. Set up Tier 2 (Nanos) above.")
            else:
                # --- Step 1: Deploy Global Server on AGX ---
                with st.expander("Step 1 — Deploy Global Server on AGX", expanded=True):
                    agx_info = AVAILABLE_DEVICES[server_ip]
                    mqw_flag  = "true" if use_mqw else "false"
                    n_nanos   = len(hfl_clusters)

                    agx_script = f"""
docker pull {image_global_server}
docker ps -aq --filter 'name=flwr-global-server' | xargs -r docker rm -f
docker run -d --name flwr-global-server \\
  --runtime=nvidia --gpus all \\
  --net=host \\
  -e SERVER_PORT={server_port} \\
  -e NUM_NANOS={n_nanos} \\
  -e GLOBAL_ROUNDS={global_rounds} \\
  -e MODEL={model} \\
  -e ALGO={algo} \\
  {image_global_server}
"""
                    st.code(agx_script, language="bash")
                    try:
                        subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{agx_info['hostname']}@{server_ip}", agx_script],
                            check=True, capture_output=True, text=True
                        )
                        st.success(f"✓ Global server deployed on {agx_info['display_name']}")
                    except subprocess.CalledProcessError as e:
                        st.error(f"✗ Failed to deploy global server")
                        with st.expander("Error"):
                            st.code(getattr(e, 'stderr', str(e)))

                # --- Step 2: Deploy Edge Aggregator on each Nano ---
                with st.expander("Step 2 — Deploy Edge Aggregators on Nanos", expanded=True):
                    for cluster in hfl_clusters:
                        n_clients = len(cluster['client_ids'])
                        nano_script = f"""
docker pull {image_edge_aggregator}
docker ps -aq --filter 'name=flwr-edge-agg' | xargs -r docker rm -f
docker run -d --name flwr-edge-agg-{cluster['cluster_idx']} \\
  --runtime=nvidia --gpus all \\
  --net=host \\
  -e CLUSTER_ID={cluster['cluster_idx']} \\
  -e EDGE_PORT={cluster['nano_port']} \\
  -e GLOBAL_SERVER_IP={server_ip} \\
  -e GLOBAL_SERVER_PORT={server_port} \\
  -e NUM_CLIENTS={n_clients} \\
  -e LOCAL_ROUNDS={local_rounds} \\
  -e USE_MQW={mqw_flag} \\
  -e MODEL={model} \\
  {image_edge_aggregator}
"""
                        st.markdown(f"**{cluster['nano_display']}** — Cluster {cluster['cluster_idx']} ({n_clients} clients)")
                        st.code(nano_script, language="bash")
                        try:
                            subprocess.run(
                                ["ssh", "-o", "StrictHostKeyChecking=no",
                                 f"{cluster['nano_hostname']}@{cluster['nano_ip']}", nano_script],
                                check=True, capture_output=True, text=True
                            )
                            st.success(f"✓ Edge aggregator deployed on {cluster['nano_display']}")
                        except subprocess.CalledProcessError as e:
                            st.error(f"✗ Failed on {cluster['nano_display']}")
                            with st.expander("Error"):
                                st.code(getattr(e, 'stderr', str(e)))

                # --- Step 3: Deploy Clients on Lambda ---
                with st.expander("Step 3 — Deploy Clients (with Nano routing)", expanded=True):
                    if not st.session_state.clients:
                        st.warning("No client devices configured in Deployment Targets.")
                    else:
                        for device in st.session_state.clients:
                            target_image = image_lambda if device['type'] == "LAMBDA" else image_edge
                            docker_flags = "--privileged --gpus all" if device['type'] == "LAMBDA" else "--runtime=nvidia --gpus all"

                            script = f"""
docker pull {target_image}
docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f
"""
                            for cid in device['client_ids']:
                                # Find which Nano this client belongs to
                                assigned_nano_ip   = server_ip    # fallback to AGX
                                assigned_nano_port = server_port
                                assigned_cluster   = 0
                                for cluster in hfl_clusters:
                                    if cid in cluster['client_ids']:
                                        assigned_nano_ip   = cluster['nano_ip']
                                        assigned_nano_port = cluster['nano_port']
                                        assigned_cluster   = cluster['cluster_idx']
                                        break

                                script += f"""
docker run -d --name flwr-client{cid} \\
  {docker_flags} \\
  --net=host \\
  --cap-add=NET_ADMIN \\
  -e CLIENT_ID={cid} \\
  -e TOTAL_CLIENTS={total_clients} \\
  -e EPOCHS={epochs} \\
  -e MODEL={model} \\
  -e ALGO={algo} \\
  -e SERVER_IP={assigned_nano_ip} \\
  -e SERVER_PORT={assigned_nano_port} \\
  -e CLUSTER_ID={assigned_cluster} \\
  -e THRESHOLD_PERCENTILE={threshold_percentile} \\
  {target_image}
"""
                            st.markdown(f"**{device['display_name']}** — {len(device['client_ids'])} clients")
                            st.code(script, language="bash")
                            try:
                                if device['ip'] in ("127.0.0.1", "localhost"):
                                    subprocess.run(script, shell=True, check=True, executable='/bin/bash')
                                else:
                                    subprocess.run(
                                        ["ssh", "-o", "StrictHostKeyChecking=no",
                                         f"{device['hostname']}@{device['ip']}", script],
                                        check=True, capture_output=True, text=True
                                    )
                                st.success(f"✓ Deployed {len(device['client_ids'])} clients to {device['display_name']}")
                            except subprocess.CalledProcessError as e:
                                st.error(f"✗ Failed on {device['display_name']}")
                                with st.expander("Error"):
                                    st.code(getattr(e, 'stderr', str(e)))

with col2:
    if st.button("🗑️ Delete All Containers", type="secondary", use_container_width=True):
        st.warning("Cleaning all configured devices...")
        
        # In HFL mode also clean Nano edge aggregators and AGX global server
        extra_filters = []
        if st.session_state.fl_architecture == "HFL":
            extra_filters = ["flwr-edge-agg", "flwr-global-server"]

        for ip, device_info in AVAILABLE_DEVICES.items():
            with st.expander(f"Cleaning {device_info['display_name']} ({ip})", expanded=True):
                filters = ["flwr-client"] + extra_filters
                for fname in filters:
                    cleanup_cmd = f"docker ps -aq --filter 'name={fname}' | xargs -r docker rm -f"
                    try:
                        if ip in ("127.0.0.1", "localhost"):
                            subprocess.run(cleanup_cmd, shell=True, check=False, executable='/bin/bash')
                        else:
                            subprocess.run(
                                ["ssh", "-o", "StrictHostKeyChecking=no",
                                 f"{device_info['hostname']}@{ip}", cleanup_cmd],
                                check=False, capture_output=True
                            )
                    except:
                        pass
                st.success(f"✓ Cleaned")

# ==========================================
# TRAINING RESULTS TABS
# ==========================================
st.markdown('<div class="section-header">Training Results</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Server Training Metrics", "🖼️ Client Training Images", "🧪 Inference Testing"])

with tab1:
    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button("📄 Fetch and Plot Server Results", use_container_width=True):
            if not server_ip:
                st.error("⚠️ No server device selected!")
            else:
                server_hostname = AVAILABLE_DEVICES[server_ip]["hostname"]
                server_display = AVAILABLE_DEVICES[server_ip]["display_name"]
                
                # Use correct container name depending on architecture
                container_name = "flwr-global-server" if st.session_state.fl_architecture == "HFL" else "flwr-server_hfl"
                
                with st.spinner(f"Fetching training logs from {server_display}..."):
                    try:
                        ssh_cmd = f"docker cp {container_name}:/app/src/log.txt /tmp/log.txt"
                        subprocess.run([
                            "ssh", "-o", "StrictHostKeyChecking=no",
                            f"{server_hostname}@{server_ip}", ssh_cmd
                        ], check=True)

                        subprocess.run([
                            "scp", f"{server_hostname}@{server_ip}:/tmp/log.txt", "log.txt"
                        ], check=True)

                        with open("log.txt", "r") as f:
                            lines = f.readlines()

                        loss_pattern = r"losses_distributed\s*(\[.*\])"
                        mape_pattern = r"metrics_distributed\s*\{.*\}"
                        train_loss_pattern = r"metrics_distributed_fit\s*\{.*\}"

                        losses, mape, train_loss = None, None, None
                        for line in lines:
                            if "losses_distributed" in line:
                                match = re.search(loss_pattern, line)
                                if match:
                                    losses = eval(match.group(1))
                            elif "metrics_distributed" in line and "metrics_distributed_fit" not in line:
                                match = re.search(mape_pattern, line)
                                if match:
                                    metrics_str = match.group(0).split("metrics_distributed")[-1].strip()
                                    mape = eval(metrics_str).get("mape", [])
                            elif "metrics_distributed_fit" in line:
                                match = re.search(train_loss_pattern, line)
                                if match:
                                    metrics_str = match.group(0).split("metrics_distributed_fit")[-1].strip()
                                    train_loss = eval(metrics_str).get("train_loss", [])

                        if not losses:
                            st.error("Could not find training results in log file.")
                        else:
                            st.session_state.losses = losses
                            st.session_state.mape = mape
                            st.session_state.train_loss = train_loss
                            st.success("✅ Successfully parsed training metrics!")
                            
                            if st.session_state.losses:
                                losses = st.session_state.losses
                                rounds_loss, loss_values = zip(*losses)

                                if st.session_state.mape:
                                    mape_list = st.session_state.mape.get('mape', []) if isinstance(st.session_state.mape, dict) else st.session_state.mape
                                    if mape_list:
                                        rounds_mape, mape_values = zip(*mape_list)
                                    else:
                                        rounds_mape, mape_values = [], []
                                else:
                                    rounds_mape, mape_values = [], []

                                rounds_train = []
                                train_values = []
                                if st.session_state.train_loss:
                                    rounds_train, train_values = zip(*st.session_state.train_loss)

                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(rounds_loss, loss_values, marker='o', color='#2a5298', label='Test Loss', linewidth=2)
                                if rounds_train:
                                    ax.plot(rounds_train, train_values, marker='^', color='#27ae60', label='Train Loss', linewidth=2, linestyle='--')
                                ax.legend(loc='upper right')
                                ax.set_xlabel("FL Round", fontsize=12)
                                ax.set_ylabel("Loss (MAE)", fontsize=12)
                                title_suffix = " [HFL - Global Rounds]" if st.session_state.fl_architecture == "HFL" else ""
                                ax.set_title(f"Training vs. Validation Loss{title_suffix}", fontsize=14, fontweight='bold')
                                ax.grid(True, linestyle='--', alpha=0.5)
                                for x, y in zip(rounds_loss, loss_values):
                                    ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, color='#2a5298')
                                if rounds_train:
                                    for x, y in zip(rounds_train, train_values):
                                        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8, color='#27ae60')
                                st.pyplot(fig)
                                
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png")
                                buf.seek(0)
                                st.download_button("💾 Download Plot as PNG", data=buf, file_name="fl_loss_plot.png", mime="image/png")

                            ssh_cmd_json = f"docker cp {container_name}:/app/src/fl_metrics.json /tmp/fl_metrics.json"
                            subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                                            f"{server_hostname}@{server_ip}", ssh_cmd_json], check=True)
                            subprocess.run(["scp", f"{server_hostname}@{server_ip}:/tmp/fl_metrics.json", "fl_metrics.json"], check=True)

                            with open("fl_metrics.json", "r") as f:
                                metrics_json = json.load(f)

                            summary  = metrics_json.get("summary", {})
                            per_round = metrics_json.get("per_round_metrics", {})

                            st.subheader("📊 Federated Learning Summary")
                            total_rounds = summary.get("total_communication_rounds", 0)
                            if per_round and total_rounds > 0:
                                avg_duration    = sum(per_round[k].get("duration_seconds", 0) for k in per_round) / total_rounds
                                avg_aggregation = sum(per_round[k].get("aggregation_time", 0)  for k in per_round) / total_rounds
                            else:
                                avg_duration = avg_aggregation = 0

                            col1, col2, col3, col4 = st.columns(4)
                            with col1: st.metric("Total Rounds", total_rounds)
                            with col2:
                                total_time = summary.get("total_training_time", 0)
                                st.metric("Total Time", f"{total_time:.2f}s", delta=f"{total_time/60:.2f} min")
                            with col3: st.metric("Avg Time/Round", f"{avg_duration:.2f}s")
                            with col4: st.metric("Avg Aggregation", f"{avg_aggregation*1000:.2f}ms")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                tb_sent = summary.get('total_bytes_sent', 0)
                                st.metric("Total Bytes Sent", f"{tb_sent/1024/1024:.2f} MB")
                            with col2:
                                tb_recv = summary.get('total_bytes_received', 0)
                                st.metric("Total Bytes Received", f"{tb_recv/1024/1024:.2f} MB")
                            with col3:
                                total_comm = summary.get("total_communication_overhead", 0)
                                st.metric("Total Communication", f"{total_comm/1024/1024:.2f} MB")
                            with col4:
                                if total_rounds > 0:
                                    st.metric("Avg Comm/Round", f"{total_comm/total_rounds/1024/1024:.2f} MB")

                            # HFL-specific: show tier breakdown if available
                            if st.session_state.fl_architecture == "HFL" and summary.get("tier1_network_overhead"):
                                st.subheader("🌿 HFL Tier Breakdown")
                                th1, th2, th3 = st.columns(3)
                                with th1: st.metric("Tier 1 Network Overhead", f"{summary.get('tier1_network_overhead', 0):.2f}s", help="Client→Nano lag")
                                with th2: st.metric("Tier 2 Network Overhead", f"{summary.get('tier2_network_overhead', 0):.2f}s", help="Nano→AGX lag")
                                with th3: st.metric("AGX Uploads/Round", summary.get("agx_uploads_per_round", len(hfl_clusters)), help="Always = #clusters, N-independent")

                            st.subheader("📋 Per-Round Metrics")
                            if per_round:
                                rounds_data = []
                                sorted_keys = sorted(per_round.keys(), key=lambda x: int(x.split("_")[1]))
                                for r_key in sorted_keys:
                                    round_num  = int(r_key.split("_")[1])
                                    round_data = per_round[r_key]
                                    row = {
                                        "Round": round_num,
                                        "Duration (s)": f"{round_data.get('duration_seconds', 0):.2f}",
                                        "Bytes Sent": f"{round_data.get('bytes_sent', 0):,}",
                                        "Bytes Received": f"{round_data.get('bytes_received', 0):,}",
                                        "Clients": round_data.get('num_clients_communicated', 0),
                                        "Aggregation Time (s)": f"{round_data.get('aggregation_time', 0):.4f}",
                                    }
                                    if st.session_state.fl_architecture == "HFL":
                                        row["Nano Uploads"] = round_data.get('nano_uploads', len(hfl_clusters))
                                    rounds_data.append(row)
                                rounds_df = pd.DataFrame(rounds_data)
                                st.dataframe(rounds_df, use_container_width=True)
                                csv = rounds_df.to_csv(index=False)
                                st.download_button("📥 Download Per-Round Metrics as CSV", data=csv,
                                                   file_name="fl_per_round_metrics.csv", mime="text/csv")

                    except subprocess.CalledProcessError as e:
                        st.error(f"Failed to fetch logs from {server_display} ({server_ip})")
                        with st.expander("Error details"):
                            st.code(e.stderr if e.stderr else str(e))

with tab2:
    st.markdown("### Fetch Training Images from Client Containers")
    if not st.session_state.clients:
        st.warning("⚠️ No client devices configured. Add devices first.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            client_options = []
            for device in st.session_state.clients:
                client_list = device.get('client_ids', range(device.get('start', 1), device.get('end', 1) + 1))
                for cid in client_list:
                    client_options.append({'label': f"{device['display_name']} - Client {cid}", 'device': device, 'client_id': cid})
            if client_options:
                selected_idx = st.selectbox("Select Client Container", range(len(client_options)),
                                             format_func=lambda x: client_options[x]['label'])
                selected_client = client_options[selected_idx]
        with col2:
            if client_options:
                st.write("")
                st.caption(f"Device: **{selected_client['device']['display_name']}**")
                st.caption(f"IP: **{selected_client['device']['ip']}**")

        if client_options:
            device = selected_client['device']
            client_id = selected_client['client_id']
            container_name = f"flwr-client{client_id}"

            try:
                list_cmd = f'docker exec {container_name} sh -c "ls /app/src/*.png 2>/dev/null || true"'
                result = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no",
                     f"{device['hostname']}@{device['ip']}", list_cmd],
                    check=True, capture_output=True, text=True
                )
                available_images = [os.path.basename(x.strip()) for x in result.stdout.splitlines() if x.strip()]
            except:
                available_images = []

            if available_images:
                image_filename = st.selectbox("Select Image File", available_images)
            else:
                st.warning("⚠️ No PNG images found in the container.")
                image_filename = st.text_input("Image Filename Pattern",
                                               value="Fedavg_LSTM_25clients_train_mape_histogram.png")

            if 'fetched_images' not in st.session_state:
                st.session_state.fetched_images = []

            if st.button("🖼️ Show Selected Image", use_container_width=True):
                with st.spinner(f"Fetching image from {device['display_name']} - Client {client_id}..."):
                    try:
                        cat_cmd = f'docker exec {container_name} cat /app/src/{image_filename}'
                        result = subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{device['hostname']}@{device['ip']}", cat_cmd],
                            check=True, capture_output=True
                        )
                        image_bytes = io.BytesIO(result.stdout)
                        img = Image.open(image_bytes)
                        st.session_state.fetched_images.append({
                            'img': img.copy(), 'from': f"{device['display_name']} - Client {client_id}",
                            'filename': image_filename
                        })
                        st.success(f"✅ Image fetched from {device['display_name']} - Client {client_id}")
                    except subprocess.CalledProcessError as e:
                        st.error("❌ Failed to fetch image from container")
                        with st.expander("Error details"):
                            st.code(e.stderr if e.stderr else str(e))

            if st.session_state.fetched_images:
                st.markdown("---")
                st.markdown("### Fetched Training Images")
                images_to_show = st.session_state.fetched_images[-2:]
                cols = st.columns(len(images_to_show))
                for idx, img_info in enumerate(images_to_show):
                    with cols[idx]:
                        st.markdown(f"**From:** {img_info['from']}")
                        st.image(img_info['img'], width=800)
                        buf = io.BytesIO()
                        img_info['img'].save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button("💾 Download Image", data=buf, file_name=img_info['filename'],
                                           mime="image/png", key=f"download_{img_info['filename']}_{idx}")

with tab3:
    st.markdown("### 🧪 Server-Side Inference Testing")
    st.info("Run anomaly detection on the server using the pre-existing script and live client thresholds.")

    c1, c2, c3 = st.columns(3)
    with c1: inf_algo = st.selectbox("Algorithm", ["FedAvg", "FedProx"], index=0)
    with c2: inf_dataset = st.selectbox("Test Dataset", ["V3S1.csv", "V3S2.csv", "V3S3.csv"])
    with c3:
        default_clients = sum([len(d.get('client_ids', [])) for d in st.session_state.clients])
        inf_clients = st.number_input("Number of Clients", min_value=1, value=max(1, default_clients))

    st.markdown("#### Threshold Configuration")
    col_auto, col_manual = st.columns([1, 3])
    with col_auto:
        if st.button("🪄 Auto-Fetch Thresholds"):
            if not st.session_state.clients:
                st.error("No clients configured!")
            else:
                fetched_thresholds = {}
                progress_text = st.empty()
                for device in st.session_state.clients:
                    c_list = device.get('client_ids', [])
                    for cid in c_list:
                        progress_text.text(f"Fetching from Client {cid}...")
                        try:
                            cmd = f"ssh -o StrictHostKeyChecking=no {device['hostname']}@{device['ip']} \"docker exec flwr-client{cid} cat /app/src/client_threshold.json\""
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            if result.returncode == 0:
                                data = json.loads(result.stdout)
                                fetched_thresholds[cid] = data['threshold_mae']
                            else:
                                st.warning(f"Client {cid}: No threshold file found.")
                        except Exception as e:
                            st.warning(f"Client {cid}: Failed ({str(e)})")
                progress_text.empty()
                if fetched_thresholds:
                    ordered_values = []
                    missing = []
                    for i in range(1, inf_clients + 1):
                        if i in fetched_thresholds:
                            ordered_values.append(str(fetched_thresholds[i]))
                        else:
                            ordered_values.append("0.05")
                            missing.append(i)
                    st.session_state.auto_threshold_str = ",".join(ordered_values)
                    if missing:
                        st.warning(f"Missing thresholds for Clients {missing}. Using default 0.05.")
                    else:
                        st.success(f"Successfully fetched {len(fetched_thresholds)} thresholds!")

    default_thresh = st.session_state.get("auto_threshold_str", "0.05," * (inf_clients - 1) + "0.05")
    threshold_input = st.text_area("Thresholds (Comma-Separated)", value=default_thresh, height=70)

    if st.button("▶️ Run Inference Test", type="primary", use_container_width=True):
        if not server_ip:
            st.error("⚠️ No server device selected in 'FL Configuration'!")
        else:
            server_host = AVAILABLE_DEVICES[server_ip]["hostname"]
            server_disp = AVAILABLE_DEVICES[server_ip]["display_name"]
            container_name = "flwr-global-server" if st.session_state.fl_architecture == "HFL" else "flwr-server_hfl"

            with st.status(f"Running inference on {server_disp}...") as status:
                try:
                    cmd_str = f"cd /app/src && python3 inference_test.py --algo {inf_algo} --dataset {inf_dataset} --clients {inf_clients} --thresholds \"{threshold_input.strip()}\""
                    ssh_run_cmd = f"docker exec {container_name} sh -c '{cmd_str}'"
                    status.write(f"Executing: {cmd_str}")

                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", f"{server_host}@{server_ip}", ssh_run_cmd],
                        capture_output=True, text=True
                    )
                    output_log = result.stdout

                    if result.returncode == 0:
                        status.write("Fetching results CSV...")
                        dataset_name = inf_dataset.replace('.csv', '')
                        csv_file = f'inference_summary_{inf_algo.lower()}_{dataset_name}_{inf_clients}clients.csv'

                        if os.path.exists(csv_file):
                            os.remove(csv_file)

                        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", f"{server_host}@{server_ip}",
                                        f"docker cp {container_name}:/app/src/{csv_file} /tmp/{csv_file}"], capture_output=True, text=True)
                        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no",
                                        f"{server_host}@{server_ip}:/tmp/{csv_file}", csv_file], capture_output=True, text=True)

                        if os.path.exists(csv_file):
                            st.success("✅ Inference Complete!")
                            df_res = pd.read_csv(csv_file)

                            avg_acc  = df_res['accuracy'].mean()
                            avg_f1   = df_res['f1_score'].mean()
                            min_f1   = df_res['f1_score'].min()
                            avg_prec = df_res['precision'].mean()
                            avg_rec  = df_res['recall'].mean()
                            f1_array = df_res['f1_score'].values
                            jain_index = (np.sum(f1_array)**2) / (len(f1_array) * np.sum(f1_array**2)) if np.sum(f1_array) > 0 else 0

                            st.markdown("#### 📊 Aggregate Performance")
                            m1, m2, m3, m4, m5, m6 = st.columns(6)
                            m1.metric("Avg F1-Score",  f"{avg_f1:.4f}")
                            m2.metric("Avg Precision", f"{avg_prec:.4f}")
                            m3.metric("Avg Recall",    f"{avg_rec:.4f}")
                            m4.metric("Avg Accuracy",  f"{avg_acc:.4f}")
                            m5.metric("Min F1 (Worst)", f"{min_f1:.4f}", delta_color="inverse")
                            m6.metric("Jain's Fairness", f"{jain_index:.4f}")

                            st.markdown("#### 📋 Per-Client Performance")
                            df_display = df_res.copy()
                            for col in ['threshold', 'precision', 'recall', 'f1_score', 'accuracy']:
                                if col in df_display.columns:
                                    df_display[col] = df_display[col].round(4)
                            st.dataframe(df_display, use_container_width=True)

                            st.markdown("#### 📈 Client Comparison")
                            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                            x_pos = np.arange(len(df_res['client_id']))
                            width = 0.2
                            ax1 = axes[0]
                            ax1.bar(x_pos - width*1.5, df_res['precision'], width, label='Precision', color='#2a5298')
                            ax1.bar(x_pos - width*0.5, df_res['recall'],    width, label='Recall',    color='#27ae60')
                            ax1.bar(x_pos + width*0.5, df_res['f1_score'],  width, label='F1-Score',  color='#e67e22')
                            ax1.bar(x_pos + width*1.5, df_res['accuracy'],  width, label='Accuracy',  color='#9b59b6')
                            ax1.set_xlabel('Client ID'); ax1.set_ylabel('Score')
                            ax1.set_title('Performance Metrics by Client', fontweight='bold')
                            ax1.set_xticks(x_pos); ax1.set_xticklabels(df_res['client_id'])
                            ax1.legend(); ax1.set_ylim([0, 1.05])

                            ax2 = axes[1]
                            ax2.bar(x_pos - width, df_res['TP'], width, label='TP', color='#27ae60')
                            ax2.bar(x_pos,          df_res['FP'], width, label='FP', color='#e74c3c')
                            ax2.bar(x_pos + width,  df_res['FN'], width, label='FN', color='#f39c12')
                            ax2.set_xlabel('Client ID'); ax2.set_ylabel('Count')
                            ax2.set_title('Detection Counts by Client', fontweight='bold')
                            ax2.set_xticks(x_pos); ax2.set_xticklabels(df_res['client_id'])
                            ax2.legend()
                            plt.tight_layout()
                            st.pyplot(fig)

                            csv_data = df_res.to_csv(index=False)
                            st.download_button("📥 Download Results (CSV)", data=csv_data,
                                               file_name=csv_file, mime="text/csv", use_container_width=True)
                        else:
                            st.error(f"⚠️ Could not fetch CSV: `{csv_file}`")
                            with st.expander("Console Output"):
                                st.code(output_log)
                    else:
                        st.error("❌ Inference Script Failed")
                        st.code(output_log)
                        if result.stderr:
                            st.code(result.stderr)

                except Exception as e:
                    st.error(f"❌ Execution Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

st.markdown("---")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 2rem;">
    <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 250px; margin-bottom: 1rem;">
            <h4 style="color: #1e3c72; margin: 0 0 0.5rem 0;">Center for Cybersecurity Research (C2SR)</h4>
            <p style="margin: 0; color: #666; font-size: 0.9rem;">University of North Dakota (UND)</p>
        </div>
        <div style="flex: 1; min-width: 250px; margin-bottom: 1rem;">
            <h4 style="color: #1e3c72; margin: 0 0 0.5rem 0;">Developer</h4>
            <p style="margin: 0; color: #666; font-size: 0.9rem;">Muhammad Hamza Karim</p>
            <p style="margin: 0; color: #666; font-size: 0.9rem;">PhD Candidate - Computer Science</p>
            <p style="margin: 0; color: #666; font-size: 0.9rem;">📧 muhammad.karim@und.edu</p>
        </div>
        <div style="flex: 1; min-width: 250px; margin-bottom: 1rem;">
            <h4 style="color: #1e3c72; margin: 0 0 0.5rem 0;">Project</h4>
            <p style="margin: 0; color: #666; font-size: 0.9rem;">Federated Learning on Edge Devices</p>
            <p style="margin: 0; color: #666; font-size: 0.9rem;">Version 3.0 (HFL) | © 2025</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)