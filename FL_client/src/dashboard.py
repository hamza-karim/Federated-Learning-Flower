import streamlit as st
import base64
import subprocess
import pandas as pd
from graphviz import Digraph
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="C2SR - FL Deployment", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for professional styling
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
    .logo-container {
        background: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3c72;
        margin: 0;
        line-height: 1.2;
    }
    .logo-subtext {
        font-size: 0.75rem;
        color: #666;
        margin: 0;
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
    .status-unknown {
        background-color: #ff9800;
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
</style>
""", unsafe_allow_html=True)

# Configuration for logo path
LOGO_PATH = "./Picture1.png" 

# Header with Logo
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #006400 0%, #00a86b 100%);
                padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0;">
            Federated Learning Deployment Platform
        </h1>
        <p style="color: #d4f8e8; font-size: 1rem; margin-top: 0.3rem;">
            Federated Learning Container Management in the C2SR Edge Testbed
        </p>
    </div>
    """, unsafe_allow_html=True)

# with col2:
#     try:
#         st.image(LOGO_PATH, use_container_width=True)
#     except:
#         st.markdown("""
#         <div style="background: white; padding: 1.5rem; border-radius: 8px; text-align: center;
#                     box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
#             <p style="font-size: 1.5rem; font-weight: 700; color: #006400; margin: 0;">C2SR</p>
#             <p style="font-size: 0.75rem; color: #666; margin: 0;">Center for Cybersecurity Research</p>
#             <p style="font-size: 0.75rem; color: #666; margin: 0;">University of North Dakota</p>
#         </div>
#         """, unsafe_allow_html=True)

with open(LOGO_PATH, "rb") as f:
    encoded_logo = base64.b64encode(f.read()).decode()

with col2:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #006400 0%, #00a86b 100%);
                padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;">
        <img src="data:image/png;base64,{encoded_logo}" 
             alt="C2SR Logo" style="width: 100%; object-fit: contain; max-height: 120px;">
    </div>
    """, unsafe_allow_html=True)



# Available devices - Add more devices here
AVAILABLE_DEVICES = {
    "10.226.47.97": "c2srnano07",
    "10.226.47.254": "c2srnano08",
    "10.226.47.85": "c2sragx04",
    "10.226.47.86": "c2srnano09",
}

# Function to check device connectivity
def check_device_status(name, ip):
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             f"{name}@{ip}", "echo 'connected'"],
            capture_output=True,
            timeout=3
        )
        return result.returncode == 0
    except:
        return False

# Dashboard Overview Section
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
    total_containers = sum(d['end'] - d['start'] + 1 for d in st.session_state.get('clients', []))
    st.metric("Total Containers", total_containers)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    st.markdown('</div>', unsafe_allow_html=True)

# Device Status Section
st.markdown('<div class="section-header">Device Status</div>', unsafe_allow_html=True)

# Initialize device status cache in session state
if 'device_status_cache' not in st.session_state:
    st.session_state.device_status_cache = {}

with st.expander("View All Available Devices", expanded=False):
    if st.button("🔄 Refresh Status", key="refresh_status"):
        # Clear cache and check all devices
        st.session_state.device_status_cache = {}
        for ip, name in AVAILABLE_DEVICES.items():
            st.session_state.device_status_cache[ip] = check_device_status(name, ip)
        st.rerun()
    
    # Only check status if cache is empty (first time)
    if not st.session_state.device_status_cache:
        with st.spinner("Checking device status..."):
            for ip, name in AVAILABLE_DEVICES.items():
                st.session_state.device_status_cache[ip] = check_device_status(name, ip)
    
    cols = st.columns(3)
    for idx, (ip, name) in enumerate(AVAILABLE_DEVICES.items()):
        with cols[idx % 3]:
            is_online = st.session_state.device_status_cache.get(ip, False)
            status_class = "status-online" if is_online else "status-offline"
            status_text = "Online" if is_online else "Offline"
            
            st.markdown(f"""
            <div class="device-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <strong>{name}</strong><br>
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

# Configuration Section
st.markdown('<div class="section-header">FL Configuration</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    # Server selection from available devices
    server_device = st.selectbox(
        "Server Device",
        options=list(AVAILABLE_DEVICES.keys()),
        format_func=lambda x: f"{AVAILABLE_DEVICES[x]} ({x})"
    )
    server_ip = server_device  # Set server_ip to selected device IP
    server_port = st.text_input("Port", "8080")
with col2:
    model = st.selectbox("Model", ["lstm", "bilstm"])
    algo = st.selectbox("Algorithm", ["fedavg", "fedprox"])
with col3:
    total_clients = st.number_input("Total Clients", min_value=1, value=4)
    epochs = st.number_input("Epochs/Client", min_value=1, max_value=50, value=5)

image = st.text_input("Docker Image", "hamzakarim07/flwr_client_hfl:latest")

st.markdown("---")

# Client Devices Section
st.markdown('<div class="section-header">Edge Device Configuration</div>', unsafe_allow_html=True)

if 'clients' not in st.session_state:
    st.session_state.clients = []

# Add device form
with st.expander("➕ Add New Device", expanded=len(st.session_state.clients)==0):
    with st.form("add_device", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            selected_ip = st.selectbox(
                "Select Device", 
                options=list(AVAILABLE_DEVICES.keys()),
                format_func=lambda x: f"{AVAILABLE_DEVICES[x]} ({x})"
            )
            start_id = st.number_input("First Client ID", min_value=1, value=1)
        with c2:
            st.write("")  # Spacer
            end_id = st.number_input("Last Client ID", min_value=1, value=1)
        
        add_btn = st.form_submit_button("Add Device to Deployment", use_container_width=True)
        if add_btn:
            if any(d['ip'] == selected_ip for d in st.session_state.clients):
                st.error(f"Device {AVAILABLE_DEVICES[selected_ip]} is already configured")
            else:
                st.session_state.clients.append({
                    "name": AVAILABLE_DEVICES[selected_ip],
                    "ip": selected_ip,
                    "start": int(start_id),
                    "end": int(end_id)
                })
                st.rerun()

# Display configured devices
if st.session_state.clients:
    st.write(f"**{len(st.session_state.clients)} device(s) configured for deployment**")
    
    for idx, device in enumerate(st.session_state.clients):
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"**{device['name']}**")
                st.caption(f"IP: {device['ip']}")
            
            with col2:
                st.write(f"Clients: {device['start']} to {device['end']}")
                st.caption(f"({device['end'] - device['start'] + 1} containers)")
            
            with col3:
                if st.button("Remove", key=f"rm_{idx}", use_container_width=True):
                    st.session_state.clients.pop(idx)
                    st.rerun()
                
                if st.button("Cleanup", key=f"clean_{idx}", use_container_width=True):
                    with st.spinner(f"Cleaning {device['name']}..."):
                        cleanup_cmd = "docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f"
                        try:
                            subprocess.run(
                                ["ssh", "-o", "StrictHostKeyChecking=no",
                                 f"{device['name']}@{device['ip']}", cleanup_cmd],
                                check=True,
                                capture_output=True
                            )
                            st.success(f"✓ Cleaned up containers on {device['name']}")
                        except subprocess.CalledProcessError:
                            st.error(f"✗ Failed to cleanup {device['name']}")
        
        st.markdown("---")
else:
    st.info("No devices configured for deployment yet. Add your first device above.")

# Network Topology
st.markdown('<div class="section-header">Network Topology</div>', unsafe_allow_html=True)

dot = Digraph(format="png")
dot.attr(rankdir='TB', bgcolor='transparent', fontname='Helvetica', fontsize='10')
dot.attr('edge', penwidth='2')

# --- Server Node ---
dot.node(
    'server',
    f"🖥️ {server_ip}:{server_port}\nFL Server\n{algo.upper()}",
    shape='box',
    style='filled,rounded,bold',
    fillcolor='#4caf50',
    fontcolor='white',
    penwidth='2'
)

# --- Client Clusters ---
for idx, device in enumerate(st.session_state.clients):
    with dot.subgraph(name=f'cluster_{idx}') as c:
        c.attr(style='rounded,dashed', color='#2a5298', label=f"{device['name']}")
        c.attr(rank='same')  # Keep all clients of a device at the same horizontal level
        
        for cid in range(device['start'], device['end']+1):
            client_node = f"{device['name']}_c{cid}"
            is_online = st.session_state.device_status_cache.get(device['ip'], True)
            fill_color = '#a8e6a1' if is_online else '#fca5a5'
            status_emoji = '🟢' if is_online else '🔴'
            
            # Client node
            c.node(
                client_node,
                f"{status_emoji} Client {cid}",
                shape='circle',
                style='filled,rounded',
                fillcolor=fill_color,
                fontcolor='black'
            )
            
            # Connect client to server
            edge_style = 'solid' if is_online else 'dashed'
            edge_color = '#2a5298' if is_online else '#ff6b6b'
            dot.edge('server', client_node, style=edge_style, color=edge_color)

# Render the enhanced topology
st.graphviz_chart(dot)

st.markdown("---")

# ------------------------------
# Training Results Visualization SERVER
# ------------------------------
st.markdown('<div class="section-header">Training Results</div>', unsafe_allow_html=True)

server_container = "flwr-server"
log_file_path = "/app/src/log.txt"  

col1, col2 = st.columns([2, 3])
with col1:
    if st.button("📄 Fetch and Plot Results", use_container_width=True):
        if not server_ip:
            st.error("⚠️ No server device selected!")
        else:
            with st.spinner(f"Fetching training logs from server {AVAILABLE_DEVICES[server_ip]}..."):
                try:
                    # SSH into the selected server device and copy log.txt locally
                    ssh_cmd = f"docker cp flwr-server:/app/src/log.txt /tmp/log.txt"
                    subprocess.run([
                        "ssh", "-o", "StrictHostKeyChecking=no",
                        f"{AVAILABLE_DEVICES[server_ip]}@{server_ip}", ssh_cmd
                    ], check=True)

                    # Now SCP it back to the dashboard host
                    subprocess.run([
                        "scp",
                        f"{AVAILABLE_DEVICES[server_ip]}@{server_ip}:/tmp/log.txt",
                        "log.txt"
                    ], check=True)

                    # Parse needed lines
                    import re
                    with open("log.txt", "r") as f:
                        lines = f.readlines()

                    loss_pattern = r"losses_distributed\s*(\[.*\])"
                    mape_pattern = r"metrics_distributed\s*\{.*\}"

                    losses, mape = None, None
                    for line in lines:
                        if "losses_distributed" in line:
                            match = re.search(loss_pattern, line)
                            if match:
                                losses = eval(match.group(1))
                        elif "metrics_distributed" in line:
                            match = re.search(mape_pattern, line)
                            if match:
                                metrics_str = match.group(0).split("metrics_distributed")[-1].strip()
                                mape = eval(metrics_str)["mape"]

                    if not losses or not mape:
                        st.error("Could not find training results in log file.")
                    else:
                        st.session_state.losses = losses
                        st.session_state.mape = mape
                        st.success("✅ Successfully parsed training metrics!")
                        # Check if losses and mape exist in session_state
                        if 'losses' in st.session_state and st.session_state.losses \
                        and 'mape' in st.session_state and st.session_state.mape:

                            # Prepare losses
                            losses = st.session_state.losses
                            rounds_loss, loss_values = zip(*losses)

                            # Prepare MAPE safely
                            if isinstance(st.session_state.mape, dict) and 'mape' in st.session_state.mape:
                                mape_list = st.session_state.mape['mape']
                            else:
                                mape_list = st.session_state.mape

                            rounds_mape, mape_values = zip(*mape_list)

                            # Print raw values
                            st.markdown("**Raw Training Metrics**")
                            st.text(f"losses_distributed {losses}")
                            st.text(f"metrics_distributed {st.session_state.mape}")

                            # Plot Loss and MAPE on same y-axis
                            fig, ax = plt.subplots(figsize=(8,5))

                            ax.plot(rounds_loss, loss_values, marker='o', color='#2a5298', label='Loss')
                            ax.plot(rounds_mape, mape_values, marker='s', color='#ff6b6b', label='MAPE')
                            ax.set_xlabel("FL Round")
                            ax.set_ylabel("Value")
                            ax.set_title("FL Training Loss and MAPE per Round")
                            ax.grid(True, linestyle='--', alpha=0.5)
                            ax.legend()

                            # Annotate values
                            for x, y in zip(rounds_loss, loss_values):
                                ax.text(x, y, f"{y:.2f}", fontsize=9, va='bottom', ha='center')
                            for x, y in zip(rounds_mape, mape_values):
                                ax.text(x, y, f"{y:.2f}", fontsize=9, va='bottom', ha='center')

                            st.pyplot(fig)

                            # Show final MAPE
                            final_mape = mape_values[-1]
                            st.metric(label="Final MAPE", value=f"{final_mape:.2f}%")

                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to fetch logs from {AVAILABLE_DEVICES[server_ip]} ({server_ip})")
                    with st.expander("Error details"):
                        st.code(e.stderr if e.stderr else str(e))
                        
# Deployment Section
st.markdown('<div class="section-header">Deployment Control</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.session_state.clients:
        if st.button("🚀 Deploy to All Devices", type="primary", use_container_width=True):
            for device in st.session_state.clients:
                with st.expander(f"Deploying to {device['name']}", expanded=True):
                    
                    script = f"""
docker pull {image}
docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f
"""
                    for cid in range(device["start"], device["end"] + 1):
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
  {image}
"""
                    
                    try:
                        result = subprocess.run(
                            ["ssh", "-o", "StrictHostKeyChecking=no",
                             f"{device['name']}@{device['ip']}", script],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        st.success(f"✓ Successfully deployed {device['end'] - device['start'] + 1} clients")
                    except subprocess.CalledProcessError as e:
                        st.error(f"✗ Deployment failed")
                        with st.expander("Error details"):
                            st.code(e.stderr if e.stderr else "Unknown error")
    else:
        st.warning("⚠️ Add at least one device to begin deployment")

with col2:
    if st.button("🗑️ Delete All Containers", type="secondary", use_container_width=True):
        st.warning("Cleaning all available devices...")
        for ip, name in AVAILABLE_DEVICES.items():
            with st.expander(f"Cleaning {name} ({ip})", expanded=True):
                cleanup_cmd = "docker ps -aq --filter 'name=flwr-client' | xargs -r docker rm -f"
                
                try:
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no",
                         f"{name}@{ip}", cleanup_cmd],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    st.success(f"✓ Removed all flwr-client containers")
                except subprocess.CalledProcessError:
                    st.error(f"✗ Cleanup failed (device may be unreachable)")

st.markdown("---")
st.caption("© 2025 Center for Cybersecurity Research (C2SR) | University of North Dakota (UND)")