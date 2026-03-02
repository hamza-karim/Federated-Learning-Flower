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

def check_device_status(hostname, ip):
    try:
        # Optimization: Don't SSH if target is localhost/lambda
        if ip == "127.0.0.1" or ip == "localhost":
            return True
            
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             f"{hostname}@{ip}", "echo 'connected'"],
            capture_output=True,
            timeout=3
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
# 6. FL CONFIGURATION
# ==========================================
st.markdown('<div class="section-header">FL Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    server_device = st.selectbox(
        "Server Device",
        options=list(AVAILABLE_DEVICES.keys()),
        format_func=lambda x: f"{AVAILABLE_DEVICES[x]['display_name']} ({x})"
    )
    server_ip = server_device
    server_port = st.text_input("Port", "8080")
with col2:
    model = st.selectbox("Model", ["lstm", "bilstm"])
    algo = st.selectbox("Algorithm", ["fedavg", "fedprox"])
with col3:
    total_clients = st.number_input("Total Clients", min_value=1, value=20) 
    epochs = st.number_input("Epochs/Client", min_value=1, max_value=50, value=5)
with col4:
    threshold_percentile = st.number_input("Anomaly Threshold (%)", min_value=90.0, max_value=100.0, value=99.0, step=0.1)

# ==========================================
# 7. DOCKER IMAGE CONFIGURATION
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

st.markdown("---")

# ==========================================
# 8. DEVICE CONFIGURATION
# ==========================================
st.markdown('<div class="section-header">Deployment Targets</div>', unsafe_allow_html=True)

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
        # Badge logic
        badge_cls = "badge-lambda" if device['type'] == "LAMBDA" else "badge-edge"
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; 
                    border: 1px solid #e0e0e0; margin-bottom: 1rem; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**{device['display_name']}** <span class='device-badge {badge_cls}'>{device['type']}</span>", unsafe_allow_html=True)
            st.caption(f"IP: {device['ip']}")
        
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
                        # Handle localhost vs SSH
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
# 9. TOPOLOGY VISUALIZATION
# ==========================================
st.markdown('<div class="section-header">Network Topology</div>', unsafe_allow_html=True)

dot = Digraph(format="png")
dot.attr(rankdir='TB', bgcolor='transparent', fontname='Helvetica', fontsize='10')
dot.attr('edge', penwidth='2')

server_display = AVAILABLE_DEVICES[server_ip]["display_name"]
dot.node(
    'server',
    f"🖥️ {server_display}\n{server_ip}:{server_port}\nFL Server\n{algo.upper()}",
    shape='box',
    style='filled,rounded,bold',
    fillcolor='#4caf50',
    fontcolor='white',
    penwidth='2'
)

for idx, device in enumerate(st.session_state.clients):
    with dot.subgraph(name=f'cluster_{idx}') as c:
        c.attr(style='rounded,dashed', color='#2a5298', label=f"{device['display_name']}")
        c.attr(rank='same')
        
        client_list = device['client_ids']
        
        for cid in client_list:
            client_node = f"{device['hostname']}_c{cid}"
            is_online = st.session_state.device_status_cache.get(device['ip'], True)
            fill_color = '#a8e6a1' if is_online else '#fca5a5'
            status_emoji = '🟢' if is_online else '🔴'
            
            c.node(
                client_node,
                f"{status_emoji} Client {cid}",
                shape='circle',
                style='filled,rounded',
                fillcolor=fill_color,
                fontcolor='black'
            )
            
            edge_style = 'solid' if is_online else 'dashed'
            edge_color = '#2a5298' if is_online else '#ff6b6b'
            dot.edge('server', client_node, style=edge_style, color=edge_color)

st.graphviz_chart(dot)

st.markdown("---")
                        
# ==========================================
# 10. DEPLOYMENT CONTROL
# ==========================================
st.markdown('<div class="section-header">Deployment Control</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.session_state.clients:
        if st.button("🚀 Deploy to All Devices", type="primary", use_container_width=True):
            for device in st.session_state.clients:
                with st.expander(f"Deploying to {device['display_name']}...", expanded=True):
                    
                    # ==========================================
                    # DYNAMIC IMAGE SELECTION LOGIC
                    # ==========================================
                    if device['type'] == "LAMBDA":
                        target_image = image_lambda
                        # LAMBDA: Needs --privileged to fix NVML/GPU access issues
                        # Uses standard --gpus all
                        docker_flags = "--privileged --gpus all"
                    else:
                        target_image = image_edge
                        # EDGE (Jetson): Needs --runtime=nvidia for JetPack
                        docker_flags = "--runtime=nvidia --gpus all"
                    
                    st.write(f"🔹 Target Image: `{target_image}`")
                    st.write(f"⚙️ Runtime Flags: `{docker_flags}`")

                    # Base script: Pull first, clean old containers
                    script = f"""
                    docker pull {target_image}
                    docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f
                    """
                    
                    client_list = device['client_ids']
                    
                    for cid in client_list:
                        # Add run command for each client
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
                        # Logic to handle Localhost (Lambda) vs Remote (SSH)
                        if device['ip'] == "127.0.0.1" or device['ip'] == "localhost":
                            # Run direct subprocess if we are ON the Lambda
                            # Using executable=/bin/bash is safer for multi-line scripts
                            subprocess.run(script, shell=True, check=True, executable='/bin/bash')
                        else:
                            # Run via SSH for remote edge devices
                            subprocess.run(
                                ["ssh", "-o", "StrictHostKeyChecking=no",
                                 f"{device['hostname']}@{device['ip']}", script],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                        
                        st.success(f"✓ Deployed {len(client_list)} containers to {device['display_name']}")
                    except subprocess.CalledProcessError as e:
                        st.error(f"✗ Deployment failed on {device['display_name']}")
                        with st.expander("Error details"):
                            st.code(e.stderr if hasattr(e, 'stderr') else str(e))
    else:
        st.warning("⚠️ Add at least one device to begin deployment")

with col2:
    if st.button("🗑️ Delete All Containers", type="secondary", use_container_width=True):
        st.warning("Cleaning all configured devices...")
        for ip, device_info in AVAILABLE_DEVICES.items():
            
            with st.expander(f"Cleaning {device_info['display_name']} ({ip})", expanded=True):
                cleanup_cmd = "docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f"
                try:
                    if ip == "127.0.0.1" or ip == "localhost":
                        subprocess.run(cleanup_cmd, shell=True, check=True, executable='/bin/bash')
                    else:
                        subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{device_info['hostname']}@{ip}", cleanup_cmd],
                            check=True, capture_output=True
                        )
                    st.success(f"✓ Removed containers")
                except subprocess.CalledProcessError:
                    st.error(f"✗ Failed (device unreachable?)")

                # Also clean server if it exists there
                cleanup_server = "docker ps -aq --filter 'name=flwr-server' | xargs -r docker rm -f"
                try:
                    if ip == "127.0.0.1" or ip == "localhost":
                        subprocess.run(cleanup_server, shell=True, check=False, executable='/bin/bash')
                    else:
                        subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{device_info['hostname']}@{ip}", cleanup_server],
                            check=False, capture_output=True
                        )
                except:
                    pass

st.markdown('<div class="section-header">Training Results</div>', unsafe_allow_html=True)

# Define Tabs
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
                
                with st.spinner(f"Fetching training logs from {server_display}..."):
                    try:
                        ssh_cmd = f"docker cp flwr-server_hfl:/app/src/log.txt /tmp/log.txt"
                        subprocess.run([
                            "ssh", "-o", "StrictHostKeyChecking=no",
                            f"{server_hostname}@{server_ip}", ssh_cmd
                        ], check=True)

                        subprocess.run([
                            "scp",
                            f"{server_hostname}@{server_ip}:/tmp/log.txt",
                            "log.txt"
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
                            
                            if 'losses' in st.session_state and st.session_state.losses:
                                losses = st.session_state.losses
                                rounds_loss, loss_values = zip(*losses)

                                if st.session_state.mape:
                                    if isinstance(st.session_state.mape, dict):
                                        mape_list = st.session_state.mape.get('mape', [])
                                    else:
                                        mape_list = st.session_state.mape
                                    if mape_list:
                                        rounds_mape, mape_values = zip(*mape_list)
                                    else:
                                        rounds_mape, mape_values = [], []
                                else:
                                    rounds_mape, mape_values = [], []

                                if st.session_state.train_loss:
                                    rounds_train, train_values = zip(*st.session_state.train_loss)
                                else:
                                    rounds_train, train_values = [], []

                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(rounds_loss, loss_values, marker='o', color='#2a5298', label='Test Loss', linewidth=2)
                                
                                if rounds_train:
                                    ax.plot(rounds_train, train_values, marker='^', color='#27ae60', label='Train Loss', linewidth=2, linestyle='--')

                                ax.legend(loc='upper right')

                                ax.set_xlabel("FL Round", fontsize=12)
                                ax.set_ylabel("Loss (MAE)", fontsize=12)
                                ax.set_title("Training vs. Validation Loss", fontsize=14, fontweight='bold')
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
                                st.download_button("💾 Download Plot as PNG", data=buf, file_name="fl_overfitting_plot.png", mime="image/png")

                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Final Test Loss", f"{loss_values[-1]:.4f}")
                                with col_b:
                                    if train_values:
                                        st.metric("Final Train Loss", f"{train_values[-1]:.4f}")
                                                    
                            ssh_cmd_json = f"docker cp flwr-server_hfl:/app/src/fl_metrics.json /tmp/fl_metrics.json"
                            subprocess.run([
                                "ssh", "-o", "StrictHostKeyChecking=no",
                                f"{server_hostname}@{server_ip}", ssh_cmd_json
                            ], check=True)

                            subprocess.run([
                                "scp",
                                f"{server_hostname}@{server_ip}:/tmp/fl_metrics.json",
                                "fl_metrics.json"
                            ], check=True)

                            with open("fl_metrics.json", "r") as f:
                                metrics_json = json.load(f)

                            summary = metrics_json.get("summary", {})
                            per_round = metrics_json.get("per_round_metrics", {})
                            
                            st.subheader("📊 Federated Learning Summary")

                            total_rounds = summary.get("total_communication_rounds", 0)
                            if per_round and total_rounds > 0:
                                avg_duration = sum(per_round[k].get("duration_seconds", 0) for k in per_round.keys()) / total_rounds
                                avg_aggregation = sum(per_round[k].get("aggregation_time", 0) for k in per_round.keys()) / total_rounds
                            else:
                                avg_duration = 0
                                avg_aggregation = 0

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Rounds", total_rounds)
                            with col2:
                                total_time = summary.get("total_training_time", 0)
                                st.metric("Total Time", f"{total_time:.2f}s", delta=f"{total_time/60:.2f} min")
                            with col3:
                                st.metric("Avg Time/Round", f"{avg_duration:.2f}s")
                            with col4:
                                st.metric("Avg Aggregation", f"{avg_aggregation*1000:.2f}ms")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                total_bytes_sent = summary.get('total_bytes_sent', 0)
                                st.metric("Total Bytes Sent", f"{total_bytes_sent/1024/1024:.2f} MB")
                            with col2:
                                total_bytes_received = summary.get('total_bytes_received', 0)
                                st.metric("Total Bytes Received", f"{total_bytes_received/1024/1024:.2f} MB")
                            with col3:
                                total_comm = summary.get("total_communication_overhead", 0)
                                st.metric("Total Communication", f"{total_comm/1024/1024:.2f} MB")
                            with col4:
                                if total_rounds > 0:
                                    avg_comm_per_round = total_comm / total_rounds
                                    st.metric("Avg Comm/Round", f"{avg_comm_per_round/1024/1024:.2f} MB")
                            
                            st.subheader("📋 Per-Round Metrics")

                            if per_round:
                                rounds_data = []
                                sorted_keys = sorted(per_round.keys(), key=lambda x: int(x.split("_")[1]))
                                
                                for r_key in sorted_keys:
                                    round_num = int(r_key.split("_")[1])
                                    round_data = per_round[r_key]
                                    rounds_data.append({
                                        "Round": round_num,
                                        "Duration (s)": f"{round_data.get('duration_seconds', 0):.2f}",
                                        "Bytes Sent": f"{round_data.get('bytes_sent', 0):,}",
                                        "Bytes Received": f"{round_data.get('bytes_received', 0):,}",
                                        "Total Bytes": f"{round_data.get('total_bytes', 0):,}",
                                        "Clients": round_data.get('num_clients_communicated', 0),
                                        "Aggregation Time (s)": f"{round_data.get('aggregation_time', 0):.4f}",
                                        "Parameters": f"{round_data.get('num_parameters', 0):,}",
                                    })
                                
                                rounds_df = pd.DataFrame(rounds_data)
                                st.dataframe(rounds_df, use_container_width=True)
                                
                                csv = rounds_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Per-Round Metrics as CSV",
                                    data=csv,
                                    file_name="fl_per_round_metrics.csv",
                                    mime="text/csv"
                                )

                                st.subheader("📈 Visualization Dashboard")

                                if per_round:
                                    rounds, durations, bytes_sent, bytes_received, total_bytes = [], [], [], [], []
                                    aggregation_times, num_clients = [], []
                                    
                                    sorted_keys = sorted(per_round.keys(), key=lambda x: int(x.split("_")[1]))
                                    
                                    for r_key in sorted_keys:
                                        round_num = int(r_key.split("_")[1])
                                        round_data = per_round[r_key]
                                        
                                        rounds.append(round_num)
                                        durations.append(round_data.get("duration_seconds", 0))
                                        bytes_sent.append(round_data.get("bytes_sent", 0))
                                        bytes_received.append(round_data.get("bytes_received", 0))
                                        total_bytes.append(round_data.get("total_bytes", 0))
                                        aggregation_times.append(round_data.get("aggregation_time", 0))
                                        num_clients.append(round_data.get("num_clients_communicated", 0))

                                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                                    fig.suptitle('Federated Learning Performance Metrics', fontsize=16, fontweight='bold')

                                    ax1.bar(rounds, durations, color='#3498db', alpha=0.8, edgecolor='black')
                                    ax1.set_xlabel("Round Number", fontsize=12, fontweight='bold')
                                    ax1.set_ylabel("Duration (seconds)", fontsize=12, fontweight='bold')
                                    ax1.set_title("Training Duration per Round", fontsize=14, fontweight='bold')
                                    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
                                    ax1.set_xticks(rounds)
                                    
                                    for i, (r, d) in enumerate(zip(rounds, durations)):
                                        ax1.text(r, d, f'{d:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

                                    width = 0.35
                                    ax2.bar(rounds, [b/1024/1024 for b in bytes_sent], width, 
                                            label='Bytes Sent', color='#2ecc71', alpha=0.8, edgecolor='black')
                                    ax2.bar(rounds, [b/1024/1024 for b in bytes_received], width, 
                                            bottom=[b/1024/1024 for b in bytes_sent],
                                            label='Bytes Received', color='#e74c3c', alpha=0.8, edgecolor='black')
                                    
                                    ax2.set_xlabel("Round Number", fontsize=12, fontweight='bold')
                                    ax2.set_ylabel("Communication (MB)", fontsize=12, fontweight='bold')
                                    ax2.set_title("Communication Overhead (Stacked)", fontsize=14, fontweight='bold')
                                    ax2.legend(loc='upper right')
                                    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
                                    ax2.set_xticks(rounds)
                                    
                                    for i, (r, t) in enumerate(zip(rounds, total_bytes)):
                                        ax2.text(r, t/1024/1024, f'{t/1024/1024:.2f}MB', 
                                                ha='center', va='bottom', fontsize=9, fontweight='bold')

                                    ax3.plot(rounds, [t*1000 for t in aggregation_times], 
                                            marker='o', linewidth=2, markersize=8, color='#9b59b6', 
                                            markerfacecolor='#e056fd', markeredgecolor='black', markeredgewidth=1.5)
                                    ax3.set_xlabel("Round Number", fontsize=12, fontweight='bold')
                                    ax3.set_ylabel("Aggregation Time (ms)", fontsize=12, fontweight='bold')
                                    ax3.set_title("Server Aggregation Time per Round", fontsize=14, fontweight='bold')
                                    ax3.grid(True, linestyle='--', alpha=0.3)
                                    ax3.set_xticks(rounds)
    
                                    for r, t in zip(rounds, aggregation_times):
                                        ax3.annotate(f'{t*1000:.2f}ms', xy=(r, t*1000), 
                                                    xytext=(0, 10), textcoords='offset points',
                                                    ha='center', fontsize=9, fontweight='bold',
                                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

                                    avg_sent = sum(bytes_sent) / len(bytes_sent)
                                    avg_received = sum(bytes_received) / len(bytes_received)
                                    
                                    labels = ['Bytes Sent\n(Server→Clients)', 'Bytes Received\n(Clients→Server)']
                                    sizes = [avg_sent, avg_received]
                                    colors = ['#2ecc71', '#e74c3c']
                                    explode = (0.05, 0.05)
                                    
                                    ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                                            autopct=lambda pct: f'{pct:.1f}%\n({pct*sum(sizes)/100/1024/1024:.2f}MB)',
                                            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
                                    ax4.set_title("Average Communication Distribution", fontsize=14, fontweight='bold')

                                    plt.tight_layout()
                                    st.pyplot(fig)

                                    buf_all = io.BytesIO()
                                    fig.savefig(buf_all, format="png", dpi=300, bbox_inches='tight')
                                    buf_all.seek(0)
                                    
                                    st.download_button(
                                        label="💾 Download Comprehensive Dashboard as PNG",
                                        data=buf_all,
                                        file_name="fl_comprehensive_metrics.png",
                                        mime="image/png"
                                    )

                                    st.subheader("📊 Additional Insights")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        fig_eff, ax_eff = plt.subplots(figsize=(8, 5))
                                        comm_efficiency = [tb/d if d > 0 else 0 for tb, d in zip(total_bytes, durations)]
                                        
                                        avg_throughput = sum(comm_efficiency) / len(comm_efficiency) if comm_efficiency else 0
                                        
                                        ax_eff.axhline(y=avg_throughput/1024, color='#e74c3c', linestyle='--', 
                                                    linewidth=3, label=f'Average: {avg_throughput/1024:.2f} KB/s', 
                                                    alpha=0.8)
                                        
                                        ax_eff.plot(rounds, [ce/1024 for ce in comm_efficiency], 
                                                    marker='o', linewidth=2, markersize=8, color='#3498db',
                                                    markerfacecolor='#5dade2', markeredgecolor='black', 
                                                    markeredgewidth=1.5, label='Per Round', alpha=0.7)
                                        
                                        ax_eff.set_xlabel("Round Number", fontsize=12, fontweight='bold')
                                        ax_eff.set_ylabel("Throughput (KB/second)", fontsize=12, fontweight='bold')
                                        ax_eff.set_title("Communication Throughput", fontsize=13, fontweight='bold')
                                        ax_eff.grid(True, linestyle='--', alpha=0.3)
                                        ax_eff.set_xticks(rounds)
                                        ax_eff.legend(loc='best')
                                        
                                        from matplotlib.ticker import FormatStrFormatter
                                        ax_eff.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                                        
                                        for r, ce in zip(rounds, comm_efficiency):
                                            ax_eff.text(r, ce/1024, f'{ce/1024:.2f}', 
                                                        ha='center', va='bottom', fontsize=8, rotation=0)
                                        
                                        ax_eff.text(rounds[-1], avg_throughput/1024, 
                                                    f' Avg: {avg_throughput/1024:.2f} KB/s',
                                                    ha='left', va='center', fontsize=10, fontweight='bold',
                                                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
                                        
                                        st.pyplot(fig_eff)
                                        
                                        st.metric(
                                            label="Average Throughput", 
                                            value=f"{avg_throughput/1024:.2f} KB/s",
                                            help="Average communication throughput across all rounds"
                                        )

                                    st.subheader("📊 Enhanced Performance Metrics")

                                    if per_round:
                                        first_round_key = sorted(per_round.keys(), key=lambda x: int(x.split("_")[1]))[0]
                                        breakdown = per_round[first_round_key].get('breakdown', {})
                                        
                                        if breakdown and breakdown.get('client_training_mean', 0) > 0:
                                            
                                            # ============================================
                                            # UPDATED ROUND 1 METRICS WITH NETWORK OVERHEAD
                                            # ============================================
                                            st.markdown("#### 📋 Round 1 System Bottlenecks")
                                            
                                            # 1. Get raw values
                                            train_max = breakdown.get('client_training_max', 0)
                                            agg_start = breakdown.get('aggregation_start_delay', 0)
                                            upload_mean = breakdown.get('client_upload_mean', 0)
                                            straggler = breakdown.get('straggler_effect', 0)
                                            
                                            # 2. Calculate the "Hidden" Network Overhead
                                            network_overhead = max(0, agg_start - train_max)
                                            
                                            # 3. Format for display
                                            upload_display = f"{upload_mean*1000:.4f}ms" if upload_mean < 0.1 else f"{upload_mean:.4f}s"
                                            overhead_display = f"{network_overhead:.2f}s"
                                            strag_display = f"{straggler*1000:.2f}ms" if straggler < 1 else f"{straggler:.2f}s"

                                            col1, col2, col3, col4 = st.columns(4)
                                            
                                            with col1:
                                                st.markdown("**Avg Training (Compute)**")
                                                st.markdown(f"<h2 style='margin:0; color:#2a5298;'>{breakdown.get('client_training_mean', 0):.2f}s</h2>", unsafe_allow_html=True)
                                                st.caption("Time spent calculating gradients")
                                            
                                            with col2:
                                                st.markdown("**Avg Straggler (Compute)**")
                                                st.markdown(f"<h2 style='margin:0; color:#e67e22;'>{strag_display}</h2>", unsafe_allow_html=True)
                                                st.caption("Waiting for slow compute devices")
                                            
                                            with col3:
                                                st.markdown("**Network Overhead (Latency)**")
                                                st.markdown(f"<h2 style='margin:0; color:#e74c3c;'>{overhead_display}</h2>", unsafe_allow_html=True)
                                                st.caption("Time wasted transferring models")
                                            
                                            with col4:
                                                st.markdown("**Avg Upload Time**")
                                                st.markdown(f"<h2 style='margin:0; color:#9b59b6;'>{upload_display}</h2>", unsafe_allow_html=True)
                                                st.caption("Pure transmission time")
                                            
                                            st.markdown("---")
                                            
                                            # ============================================
                                            # Aggregate Across All Rounds
                                            # ============================================
                                            all_sync_eff = []
                                            all_straggler = []
                                            all_training_mean = []
                                            all_network_overhead = []
                                            
                                            for r_key in sorted(per_round.keys(), key=lambda x: int(x.split("_")[1])):
                                                bd = per_round[r_key].get('breakdown', {})
                                                if bd.get('client_training_mean', 0) > 0:
                                                    all_sync_eff.append(bd['synchronization_efficiency'])
                                                    all_straggler.append(bd['straggler_effect'])
                                                    all_training_mean.append(bd['client_training_mean'])
                                                    
                                                    # Calculate Network Overhead for this round
                                                    r_train_max = bd.get('client_training_max', 0)
                                                    r_agg_start = bd.get('aggregation_start_delay', 0)
                                                    all_network_overhead.append(max(0, r_agg_start - r_train_max))
                                            
                                            if all_sync_eff and len(all_sync_eff) > 1:
                                                st.markdown("#### 📊 Average Across All Rounds")
                                                
                                                col_a, col_b, col_c, col_d = st.columns(4)
                                                
                                                with col_a:
                                                    avg_sync = np.mean(all_sync_eff) * 100
                                                    st.markdown("**Avg Sync Efficiency**")
                                                    st.markdown(f"<h2 style='margin:0; color:#27ae60;'>{avg_sync:.1f}%</h2>", unsafe_allow_html=True)
                                                    st.caption(f"Range: {min(all_sync_eff)*100:.1f}% - {max(all_sync_eff)*100:.1f}%")
                                                
                                                with col_b:
                                                    avg_strag = np.mean(all_straggler)
                                                    strag_display = f"{avg_strag*1000:.2f}ms" if avg_strag < 1 else f"{avg_strag:.2f}s"
                                                    st.markdown("**Avg Compute Straggler**")
                                                    st.markdown(f"<h2 style='margin:0; color:#e67e22;'>{strag_display}</h2>", unsafe_allow_html=True)
                                                
                                                with col_c:
                                                    avg_overhead = np.mean(all_network_overhead)
                                                    overhead_display = f"{avg_overhead:.2f}s"
                                                    st.markdown("**Avg Network Overhead**")
                                                    st.markdown(f"<h2 style='margin:0; color:#e74c3c;'>{overhead_display}</h2>", unsafe_allow_html=True)
                                                    st.caption("Major bottleneck indicator")

                                                with col_d:
                                                    avg_training = np.mean(all_training_mean)
                                                    st.markdown("**Avg Client Training**")
                                                    st.markdown(f"<h2 style='margin:0; color:#2a5298;'>{avg_training:.2f}s</h2>", unsafe_allow_html=True)
                                                
                                                st.markdown("---")
                                                
                                                # ============================================
                                                # Visualization
                                                # ============================================
                                                st.markdown("#### 📈 Efficiency Trends")
                                                
                                                rounds_list = []
                                                sync_eff_list = []
                                                overhead_list = []
                                                
                                                for r_key in sorted(per_round.keys(), key=lambda x: int(x.split("_")[1])):
                                                    round_num = int(r_key.split("_")[1])
                                                    bd = per_round[r_key].get('breakdown', {})
                                                    if bd.get('client_training_mean', 0) > 0:
                                                        rounds_list.append(round_num)
                                                        sync_eff_list.append(bd['synchronization_efficiency'] * 100)
                                                        r_train_max = bd.get('client_training_max', 0)
                                                        r_agg_start = bd.get('aggregation_start_delay', 0)
                                                        overhead_list.append(max(0, r_agg_start - r_train_max))
                                                
                                                if rounds_list:
                                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                                                    
                                                    # Plot 1: Sync Efficiency
                                                    ax1.plot(rounds_list, sync_eff_list, marker='o', linewidth=2, 
                                                            color='#2a5298', markersize=8, markerfacecolor='#5dade2')
                                                    ax1.axhline(y=80, color='orange', linestyle='--', linewidth=2, 
                                                                label='Target: 80%', alpha=0.7)
                                                    ax1.set_xlabel("Round Number", fontsize=12, fontweight='bold')
                                                    ax1.set_ylabel("Efficiency (%)", fontsize=12, fontweight='bold')
                                                    ax1.set_title("Synchronization Efficiency", fontsize=14, fontweight='bold')
                                                    ax1.set_ylim([0, 105])
                                                    ax1.grid(True, alpha=0.3)
                                                    ax1.legend()
                                                    
                                                    # Plot 2: Network Overhead (THE NEW METRIC)
                                                    ax2.plot(rounds_list, overhead_list, marker='s', linewidth=2, 
                                                            color='#e74c3c', markersize=8, markerfacecolor='#f39c12')
                                                    ax2.set_xlabel("Round Number", fontsize=12, fontweight='bold')
                                                    ax2.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
                                                    ax2.set_title("Network Overhead (System Lag)", fontsize=14, fontweight='bold')
                                                    ax2.grid(True, alpha=0.3)
                                                    
                                                    for r, o in zip(rounds_list, overhead_list):
                                                        ax2.text(r, o, f'{o:.1f}s', ha='center', va='bottom', fontsize=9)
                                                    
                                                    plt.tight_layout()
                                                    st.pyplot(fig)
                                                    
                                                    # Download button
                                                    buf = io.BytesIO()
                                                    fig.savefig(buf, format="png", dpi=300)
                                                    buf.seek(0)
                                                    st.download_button(
                                                        "💾 Download Efficiency Analysis",
                                                        data=buf,
                                                        file_name="efficiency_analysis.png",
                                                        mime="image/png",
                                                        use_container_width=True
                                                    )
                                        else:
                                            st.info("ℹ️ Enhanced metrics not available. Make sure clients are reporting training times correctly.")

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
                if 'client_ids' in device:
                    client_list = device['client_ids']
                else:
                    client_list = range(device['start'], device['end'] + 1)
                
                for cid in client_list:
                    client_options.append({
                        'label': f"{device['display_name']} - Client {cid}",
                        'device': device,
                        'client_id': cid
                    })
            
            if client_options:
                selected_idx = st.selectbox(
                    "Select Client Container",
                    range(len(client_options)),
                    format_func=lambda x: client_options[x]['label']
                )
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
            except subprocess.CalledProcessError:
                available_images = []

            if available_images:
                image_filename = st.selectbox(
                    "Select Image File",
                    available_images,
                    help="Select a PNG image file from the live container"
                )
            else:
                st.warning("⚠️ No PNG images found in the container.")
                image_filename = st.text_input(
                    "Image Filename Pattern",
                    value="Fedavg_LSTM_25clients_train_mape_histogram.png",
                    help="Enter the PNG file name manually"
                )
            
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
                            'img': img.copy(),
                            'from': f"{device['display_name']} - Client {client_id}",
                            'filename': image_filename
                        })
                        st.success(f"✅ Image fetched successfully from {device['display_name']} - Client {client_id}")

                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Failed to fetch image from container")
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

                        st.download_button(
                            label=f"💾 Download Image",
                            data=buf,
                            file_name=img_info['filename'],
                            mime="image/png",
                            key=f"download_{img_info['filename']}_{idx}"
                        )


# ==============================================================================
# TAB 4: INFERENCE TESTING
# ==============================================================================
with tab3:
    st.markdown("### 🧪 Server-Side Inference Testing")
    st.info("Run anomaly detection on the server using the pre-existing script and live client thresholds.")

    # 1. Test Configuration
    c1, c2, c3 = st.columns(3)
    with c1:
        inf_algo = st.selectbox("Algorithm", ["FedAvg", "FedProx"], index=0)
    with c2:
        inf_dataset = st.selectbox("Test Dataset", ["V3S1.csv", "V3S2.csv", "V3S3.csv"])
    with c3:
        # Default to the number of configured clients
        default_clients = sum([len(d.get('client_ids', [])) if 'client_ids' in d else (d['end'] - d['start'] + 1) for d in st.session_state.clients])
        inf_clients = st.number_input("Number of Clients", min_value=1, value=max(1, default_clients))

    st.markdown("#### Threshold Configuration")
    
    # 2. Auto-Fetch Logic
    col_auto, col_manual = st.columns([1, 3])
    with col_auto:
        if st.button("🪄 Auto-Fetch Thresholds", help="Pull latest thresholds directly from client containers"):
            if not st.session_state.clients:
                st.error("No clients configured!")
            else:
                fetched_thresholds = {}
                progress_text = st.empty()
                
                for device in st.session_state.clients:
                    if 'client_ids' in device:
                        c_list = device['client_ids']
                    else:
                        c_list = range(device['start'], device['end'] + 1)
                    
                    for cid in c_list:
                        progress_text.text(f"Fetching from Client {cid}...")
                        try:
                            # Read JSON directly from client container
                            cmd = f"ssh -o StrictHostKeyChecking=no {device['hostname']}@{device['ip']} \"docker exec flwr-client{cid} cat /app/src/client_threshold.json\""
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                data = json.loads(result.stdout)
                                fetched_thresholds[cid] = data['threshold_mae']
                            else:
                                st.warning(f"Client {cid}: No threshold file found.")
                        except Exception as e:
                            st.warning(f"Client {cid}: Failed to fetch ({str(e)})")
                
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
    
    # 3. Threshold Display/Edit
    default_thresh = st.session_state.get("auto_threshold_str", "0.05," * (inf_clients-1) + "0.05")
    threshold_input = st.text_area("Thresholds (Comma-Separated)", value=default_thresh, height=70, 
                                   help="Client 1, Client 2, Client 3...")

# 4. Run Inference
    if st.button("▶️ Run Inference Test", type="primary", use_container_width=True):
        if not server_ip:
            st.error("⚠️ No server device selected in 'FL Configuration'!")
        else:
            server_host = AVAILABLE_DEVICES[server_ip]["hostname"]
            server_disp = AVAILABLE_DEVICES[server_ip]["display_name"]
            
            with st.status(f"Running inference on {server_disp}...") as status:
                try:
                    # Execute inference script
                    cmd_str = f"cd /app/src && python3 inference_test.py --algo {inf_algo} --dataset {inf_dataset} --clients {inf_clients} --thresholds \"{threshold_input.strip()}\""
                    ssh_run_cmd = f"docker exec flwr-server_hfl sh -c '{cmd_str}'"
                    
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
                        
                        # === FIX 1: Delete stale local files to prevent mismatched data ===
                        if os.path.exists(csv_file):
                            os.remove(csv_file)
                        
                        # Fetch from Docker
                        cp_res = subprocess.run([
                            "ssh", "-o", "StrictHostKeyChecking=no", f"{server_host}@{server_ip}", 
                            f"docker cp flwr-server_hfl:/app/src/{csv_file} /tmp/{csv_file}"
                        ], capture_output=True, text=True)
                        
                        # Fetch to local dashboard
                        scp_res = subprocess.run([
                            "scp", "-o", "StrictHostKeyChecking=no", 
                            f"{server_host}@{server_ip}:/tmp/{csv_file}", csv_file
                        ], capture_output=True, text=True)
                        
                        if os.path.exists(csv_file):
                            st.success("✅ Inference Complete!")
                            df_res = pd.read_csv(csv_file)
                            
                            # === FIX 2: Calculate metrics directly from the Table data ===
                            # This guarantees the top numbers will ALWAYS match the table below
                            avg_acc = df_res['accuracy'].mean()
                            avg_f1 = df_res['f1_score'].mean()
                            min_f1 = df_res['f1_score'].min()
                            avg_prec = df_res['precision'].mean()
                            avg_rec = df_res['recall'].mean()
                            
                            # Jain's Fairness Index calculated on F1 scores
                            f1_array = df_res['f1_score'].values
                            jain_index = (np.sum(f1_array)**2) / (len(f1_array) * np.sum(f1_array**2)) if np.sum(f1_array) > 0 else 0
                            
                            # ============================================
                            # Display AGGREGATE metrics (100% matched to table)
                            # ============================================
                            st.markdown("#### 📊 Aggregate Performance (Calculated from Table Data)")
                            m1, m2, m3, m4, m5, m6 = st.columns(6)                
                            m1.metric("Avg F1-Score", f"{avg_f1:.4f}")
                            m2.metric("Avg Precision", f"{avg_prec:.4f}")
                            m3.metric("Avg Recall", f"{avg_rec:.4f}")
                            m4.metric("Avg Accuracy", f"{avg_acc:.4f}")
                            m5.metric("Min F1 (Worst)", f"{min_f1:.4f}", delta_color="inverse")
                            m6.metric("Jain's Fairness", f"{jain_index:.4f}", help="1.0 = perfect fairness")
                            
                            # ============================================
                            # Display PER-CLIENT results table
                            # ============================================
                            st.markdown("#### 📋 Per-Client Performance")
                            
                            df_display = df_res.copy()
                            numeric_cols = ['threshold', 'precision', 'recall', 'f1_score', 'accuracy']
                            if 'mae_mean' in df_display.columns:
                                numeric_cols.extend(['mae_mean', 'mae_median', 'mae_max'])
                            
                            for col in numeric_cols:
                                if col in df_display.columns:
                                    df_display[col] = df_display[col].round(4)
                            
                            st.dataframe(
                                df_display, 
                                use_container_width=True,
                                column_config={
                                    "client_id": st.column_config.NumberColumn("Client ID", format="%d"),
                                    "threshold": st.column_config.NumberColumn("Threshold (MAE)", format="%.6f"),
                                    "TP": st.column_config.NumberColumn("True Positives", format="%d"),
                                    "FP": st.column_config.NumberColumn("False Positives", format="%d"),
                                    "TN": st.column_config.NumberColumn("True Negatives", format="%d"),
                                    "FN": st.column_config.NumberColumn("False Negatives", format="%d"),
                                }
                            )
                            
                            # ============================================
                            # Visualization: Compare clients
                            # ============================================
                            st.markdown("#### 📈 Client Comparison")
                            plt.style.use('seaborn-v0_8-darkgrid')
                            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                            
                            # Plot 1: Metrics comparison
                            ax1 = axes[0]
                            x = df_res['client_id']
                            width = 0.2
                            x_pos = np.arange(len(x))
                            
                            ax1.bar(x_pos - width*1.5, df_res['precision'], width, label='Precision', color='#2a5298')
                            ax1.bar(x_pos - width*0.5, df_res['recall'], width, label='Recall', color='#27ae60')
                            ax1.bar(x_pos + width*0.5, df_res['f1_score'], width, label='F1-Score', color='#e67e22')
                            ax1.bar(x_pos + width*1.5, df_res['accuracy'], width, label='Accuracy', color='#9b59b6')
                            
                            ax1.set_xlabel('Client ID', fontsize=12, fontweight='bold')
                            ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
                            ax1.set_title('Performance Metrics by Client', fontsize=14, fontweight='bold')
                            ax1.set_xticks(x_pos)
                            ax1.set_xticklabels(x)
                            ax1.legend(loc='lower right', frameon=True)
                            ax1.set_ylim([0, 1.05])
                            
                            # Plot 2: Confusion matrix metrics
                            ax2 = axes[1]
                            ax2.bar(x_pos - width, df_res['TP'], width, label='TP', color='#27ae60')
                            ax2.bar(x_pos, df_res['FP'], width, label='FP', color='#e74c3c')
                            ax2.bar(x_pos + width, df_res['FN'], width, label='FN', color='#f39c12')
                            
                            ax2.set_xlabel('Client ID', fontsize=12, fontweight='bold')
                            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
                            ax2.set_title('Detection Counts by Client', fontsize=14, fontweight='bold')
                            ax2.set_xticks(x_pos)
                            ax2.set_xticklabels(x)
                            ax2.legend(loc='upper right', frameon=True)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Download button
                            csv_data = df_res.to_csv(index=False)
                            st.download_button(
                                "📥 Download Detailed Results (CSV)", 
                                data=csv_data, file_name=csv_file, mime="text/csv", use_container_width=True
                            )
                        else:
                            # === FIX 3: Transparent Error Reporting ===
                            st.error(f"⚠️ Could not fetch CSV file from Server: `{csv_file}`")
                            st.info("The table from your previous run was safely deleted to prevent mismatched data. Please verify that `inference_test.py` is generating the exact filename requested above.")
                            with st.expander("Show Console Output Logs"):
                                st.code(output_log, language="text")
                                if cp_res.stderr:
                                    st.error(f"Copy Error: {cp_res.stderr}")
                            
                    else:
                        st.error("❌ Inference Script Failed")
                        st.code(output_log, language="text")
                        if result.stderr:
                            st.error("STDERR:")
                            st.code(result.stderr, language="text")

                except Exception as e:
                    st.error(f"❌ Execution Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

st.markdown("---")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
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
            <p style="margin: 0; color: #666; font-size: 0.9rem;">Version 2.0 (Hybrid) | © 2025</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)