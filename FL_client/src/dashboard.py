import os
import io
import re
import json
import base64
import subprocess
import pandas as pd
import streamlit as st
import PIL.Image as Image
from graphviz import Digraph
from datetime import datetime
import matplotlib.pyplot as plt


st.set_page_config(page_title="C2SR - FL Deployment", layout="wide", initial_sidebar_state="collapsed")

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
    .status-online {
        background-color: #4caf50;
    }
    .status-offline {
        background-color: #f44336;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .device-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin-bottom: 0.8rem;
    }
    .stExpander {
        background: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .stTextInput > div > div > input {
        border-radius: 6px;
    }
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    div[data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

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

AVAILABLE_DEVICES = {
    "10.226.44.86": {
        "hostname": "c2sragx04",
        "display_name": "AGX 04"
    },
    "10.226.47.0": {
        "hostname": "c2srnano07",
        "display_name": "Nano 07"
    },
    "10.226.47.108": {
        "hostname": "c2srnano08",
        "display_name": "Nano 08"
    },
    "10.226.46.8": {
        "hostname": "hamzakarim",
        "display_name": "Nano 10"
    },
    "10.226.47.64": {
        "hostname": "hamzakarim",
        "display_name": "Nano 13"
    },
}

def check_device_status(hostname, ip):
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             f"{hostname}@{ip}", "echo 'connected'"],
            capture_output=True,
            timeout=3
        )
        return result.returncode == 0
    except:
        return False

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
            
            st.markdown(f"""
            <div class="device-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>{device_info["display_name"]}</strong><br>
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
    total_clients = st.number_input("Total Clients", min_value=1, value=4)
    epochs = st.number_input("Epochs/Client", min_value=1, max_value=50, value=5)
with col4:
    threshold_percentile = st.number_input("Anomaly Threshold (%)", min_value=90.0, max_value=100.0, value=99.0, step=0.1)

image = st.text_input("Docker Image", "hamzakarim07/flwr_client_hfl:latest")

st.markdown("---")

st.markdown('<div class="section-header">Edge Device Configuration</div>', unsafe_allow_html=True)

if 'clients' not in st.session_state:
    st.session_state.clients = []

with st.expander("➕ Add New Device", expanded=len(st.session_state.clients) == 0):
    with st.form("add_device", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            selected_ip = st.selectbox(
                "Select Device",
                options=list(AVAILABLE_DEVICES.keys()),
                format_func=lambda x: f"{AVAILABLE_DEVICES[x]['display_name']} ({x})"
            )
        with c2:
            client_range = st.text_input(
                "Client IDs (e.g., 1-3, 5, 7-9)", 
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
                elif any(d['ip'] == selected_ip for d in st.session_state.clients):
                    st.error(f"Device {AVAILABLE_DEVICES[selected_ip]['display_name']} is already configured")
                else:
                    st.session_state.clients.append({
                        "hostname": AVAILABLE_DEVICES[selected_ip]["hostname"],
                        "display_name": AVAILABLE_DEVICES[selected_ip]["display_name"],
                        "ip": selected_ip,
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
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; 
                    border: 1px solid #e0e0e0; margin-bottom: 1rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**{device['display_name']}**")
            st.caption(f"IP: {device['ip']}")
        
        with col2:
            if 'client_ids' in device:
                ids_str = ', '.join(map(str, device['client_ids']))
                st.write(f"Clients: {ids_str}")
                st.caption(f"({len(device['client_ids'])} containers)")
            else:
                st.write(f"Clients: {device['start']} to {device['end']}")
                st.caption(f"({device['end'] - device['start'] + 1} containers)")
        
        with col3:
            if st.button("Remove", key=f"rm_{idx}", use_container_width=True):
                st.session_state.clients.pop(idx)
                st.rerun()
            
            if st.button("Cleanup", key=f"clean_{idx}", use_container_width=True):
                with st.spinner(f"Cleaning {device['display_name']}..."):
                    cleanup_cmd = "docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f"
                    try:
                        subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{device['hostname']}@{device['ip']}", cleanup_cmd],
                            check=True,
                            capture_output=True
                        )
                        st.success(f"✓ Cleaned up containers on {device['display_name']}")
                    except subprocess.CalledProcessError:
                        st.error(f"✗ Failed to cleanup {device['display_name']}")
        
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No devices configured for deployment yet. Add your first device above.")

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
        
        if 'client_ids' in device:
            client_list = device['client_ids']
        else:
            client_list = range(device['start'], device['end'] + 1)
        
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
                        
st.markdown('<div class="section-header">Deployment Control</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.session_state.clients:
        if st.button("🚀 Deploy to All Devices", type="primary", use_container_width=True):
            for device in st.session_state.clients:
                with st.expander(f"Deploying to {device['display_name']}", expanded=True):
                    
                    script = f"""
docker pull {image}
docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f
"""
                    if 'client_ids' in device:
                        client_list = device['client_ids']
                    else:
                        client_list = range(device['start'], device['end'] + 1)
                    
                    for cid in client_list:
                        script += f"""
docker run -d --name flwr-client{cid} \\
  --runtime=nvidia --gpus all \\
  -e CLIENT_ID={cid} \\
  -e TOTAL_CLIENTS={total_clients} \\
  -e EPOCHS={epochs} \\
  -e MODEL={model} \\
  -e ALGO={algo} \\
  -e SERVER_IP={server_ip} \\
  -e SERVER_PORT={server_port} \\
  -e THRESHOLD_PERCENTILE={threshold_percentile} \\
  {image}
"""
                    try:
                        result = subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{device['hostname']}@{device['ip']}", script],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        num_deployed = len(client_list) if 'client_ids' in device else device['end'] - device['start'] + 1
                        st.success(f"✓ Successfully deployed {num_deployed} clients")
                    except subprocess.CalledProcessError as e:
                        st.error(f"✗ Deployment failed")
                        with st.expander("Error details"):
                            st.code(e.stderr if e.stderr else "Unknown error")
    else:
        st.warning("⚠️ Add at least one device to begin deployment")

with col2:
    if st.button("🗑️ Delete All Containers", type="secondary", use_container_width=True):
        st.warning("Cleaning all available devices...")
        for ip, device_info in AVAILABLE_DEVICES.items():
            with st.expander(f"Cleaning {device_info['display_name']} ({ip})", expanded=True):
                cleanup_cmd = "docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f"
                try:
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no",
                         f"{device_info['hostname']}@{ip}", cleanup_cmd],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    st.success(f"✓ Removed all flwr-client containers")
                except subprocess.CalledProcessError:
                    st.error(f"✗ Cleanup failed (device may be unreachable)")

                cleanup_server_cmd = "docker ps -aq --filter 'name=flwr-server_hfl' | xargs -r docker rm -f"
                try:
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no",
                         f"{device_info['hostname']}@{ip}", cleanup_server_cmd],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    st.success(f"✓ Removed flwr-server_hfl container")
                except subprocess.CalledProcessError:
                    st.error(f"✗ flwr-server_hfl cleanup failed (device may be unreachable)")

st.markdown('<div class="section-header">Training Results</div>', unsafe_allow_html=True)

server_container = "flwr-server_hfl"
log_file_path = "/app/src/log.txt"  

tab1, tab2, tab3, tab4 = st.tabs(["📊 Server Training Metrics", "🖼️ Client Training Images", "📋 Client Thresholds", "🧪 Inference Testing"])

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

                                if rounds_mape:
                                    ax2 = ax.twinx()
                                    ax2.plot(rounds_mape, mape_values, marker='s', color='#ff6b6b', label='Test MAPE', linewidth=1.5, alpha=0.7)
                                    ax2.set_ylabel("MAPE (%)", color='#ff6b6b', fontsize=12)
                                    ax2.tick_params(axis='y', labelcolor='#ff6b6b')
                                    lines_1, labels_1 = ax.get_legend_handles_labels()
                                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                                    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
                                else:
                                    ax.legend(loc='upper right')

                                ax.set_xlabel("FL Round", fontsize=12)
                                ax.set_ylabel("Loss (MAE)", fontsize=12)
                                ax.set_title("Training vs. Validation Loss (Overfitting Check)", fontsize=14, fontweight='bold')
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

                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Final Test Loss", f"{loss_values[-1]:.4f}")
                                with col_b:
                                    if train_values:
                                        st.metric("Final Train Loss", f"{train_values[-1]:.4f}")
                                with col_c:
                                    if mape_values:
                                        st.metric("Final Test MAPE", f"{mape_values[-1]:.2f}%")
                                                    
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

with tab3:
    st.markdown("### 📋 Client Thresholds")
    
    if st.button("🔄 Fetch All Client Thresholds", type="primary", use_container_width=True):
        if not st.session_state.clients:
            st.warning("⚠️ No devices configured.")
        else:
            threshold_data_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_containers = 0
            for device in st.session_state.clients:
                if 'client_ids' in device:
                    total_containers += len(device['client_ids'])
                else:
                    total_containers += (device['end'] - device['start'] + 1)
            
            processed_count = 0
            
            for device in st.session_state.clients:
                if 'client_ids' in device:
                    client_list = device['client_ids']
                else:
                    client_list = range(device['start'], device['end'] + 1)
                
                for cid in client_list:
                    status_text.text(f"Fetching threshold from {device['display_name']} - Client {cid}...")
                    container_name = f"flwr-client{cid}"
                    
                    try:
                        cat_cmd = f'docker exec {container_name} cat /app/src/client_threshold.json'
                        result = subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{device['hostname']}@{device['ip']}", cat_cmd],
                            capture_output=True, text=True
                        )
                        
                        if result.returncode == 0:
                            data = json.loads(result.stdout)
                            data['device_name'] = device['display_name']
                            data['ip'] = device['ip']
                            threshold_data_list.append(data)
                        else:
                            threshold_data_list.append({
                                'client_id': cid,
                                'device_name': device['display_name'],
                                'threshold_mape': None,
                                'note': 'File not found'
                            })
                            
                    except Exception as e:
                        threshold_data_list.append({
                            'client_id': cid,
                            'device_name': device['display_name'],
                            'threshold_mape': None,
                            'note': f'Error: {str(e)}'
                        })
                    
                    processed_count += 1
                    progress_bar.progress(processed_count / total_containers)
            
            status_text.empty()
            progress_bar.empty()
            
            if threshold_data_list:
                df_thresholds = pd.DataFrame(threshold_data_list)
                cols = ['client_id', 'device_name', 'threshold_mape', 'threshold_percentile', 'train_mape_mean', 'test_mape_mean']
                cols = [c for c in cols if c in df_thresholds.columns]
                if 'note' in df_thresholds.columns:
                    cols.append('note')
                    
                st.dataframe(df_thresholds[cols], use_container_width=True)
                
                csv = df_thresholds.to_csv(index=False)
                st.download_button(
                    label="📥 Download Thresholds CSV",
                    data=csv,
                    file_name="client_thresholds.csv",
                    mime="text/csv"
                )
            else:
                st.info("No threshold data collected.")

# ==============================================================================
# TAB 4: INFERENCE TESTING (Fixed Path)
# ==============================================================================
with tab4:
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
                                fetched_thresholds[cid] = data['threshold_mape']
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
                    # FIX: Use absolute path /app/src/inference_test.py
                    # Also added 'cd /app/src &&' to ensure relative paths inside the script work
                    cmd_str = f"cd /app/src && python3 inference_test.py --algo {inf_algo} --dataset {inf_dataset} --clients {inf_clients} --thresholds \"{threshold_input.strip()}\""
                    
                    ssh_run_cmd = f"docker exec flwr-server_hfl sh -c '{cmd_str}'"
                    
                    status.write(f"Executing: {cmd_str}")
                    
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", f"{server_host}@{server_ip}", ssh_run_cmd],
                        capture_output=True, text=True
                    )
                    
                    output_log = result.stdout
                    csv_file = f"inference_summary_{inf_algo}_{inf_clients}clients.csv"
                    
                    jain_index, avg_acc, avg_f1, avg_prec, avg_rec = "N/A", "N/A", "N/A", "N/A", "N/A"
                    if "JAIN_INDEX:" in output_log:
                        jain_index = output_log.split("JAIN_INDEX:")[1].split()[0]
                    if "AVG_ACC:" in output_log:
                        avg_acc = output_log.split("AVG_ACC:")[1].split()[0]
                    if "AVG_F1:" in output_log:
                        avg_f1 = output_log.split("AVG_F1:")[1].split()[0]
                    if "AVG_PREC:" in output_log:
                        avg_prec = output_log.split("AVG_PREC:")[1].split()[0] 
                    if "AVG_REC:" in output_log:
                        avg_rec = output_log.split("AVG_REC:")[1].split()[0]    
                           
                    if result.returncode == 0:
                        status.write("Fetching results CSV...")
                        subprocess.run([
                            "ssh", "-o", "StrictHostKeyChecking=no", f"{server_host}@{server_ip}", 
                            f"docker cp flwr-server_hfl:/app/src/{csv_file} /tmp/{csv_file}"
                        ])
                        subprocess.run([
                            "scp", "-o", "StrictHostKeyChecking=no", 
                            f"{server_host}@{server_ip}:/tmp/{csv_file}", csv_file
                        ])
                        
                        st.success("Inference Complete!")
                        
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Jain's Fairness Index", jain_index)
                        m2.metric("Avg Accuracy", avg_acc)
                        m3.metric("Avg F1-Score", avg_f1)
                        m4.metric("Avg Precision", avg_prec)
                        m5.metric("Avg Recall", avg_rec)
                        
                        
                        if os.path.exists(csv_file):
                            df_res = pd.read_csv(csv_file)
                            st.dataframe(df_res, use_container_width=True)
                            
                            csv_data = df_res.to_csv(index=False)
                            st.download_button("📥 Download Inference Results", data=csv_data, file_name=csv_file, mime="text/csv")
                    else:
                        st.error("Inference Script Failed")
                        st.error(output_log)
                        if result.stderr:
                            st.code(result.stderr)

                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")

st.markdown("---")

# Footer with detailed information
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
            <p style="margin: 0; color: #666; font-size: 0.9rem;">Version 1.0 | © 2025</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)