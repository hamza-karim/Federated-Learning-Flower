import streamlit as st
import subprocess
from graphviz import Digraph

st.set_page_config(page_title="FL Client Deployment", layout="wide")

# Custom CSS for more natural styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e8e8e8;
        padding-bottom: 0.5rem;
    }
    .device-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 0.8rem;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Federated Learning Deployment</div>', unsafe_allow_html=True)
st.write("Manage and deploy FL clients across edge devices")

st.markdown("---")

# Configuration Section
st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    server_ip = st.text_input("Server Address", "10.226.47.97")
    server_port = st.text_input("Port", "8080")
with col2:
    model = st.selectbox("Model", ["lstm", "bilstm"])
    algo = st.selectbox("Algorithm", ["fedavg", "fedprox"])
with col3:
    total_clients = st.number_input("Total Clients", min_value=1, value=4)
    epochs = st.number_input("Epochs/Client", min_value=1, max_value=50, value=5)

image = st.text_input("Docker Image", "hamzakarim07/flwr_client_hfl:latest")

st.markdown("---")

# Available devices - Add more devices here
AVAILABLE_DEVICES = {
    "10.226.47.97": "c2srnano07",
    "10.226.47.254": "c2srnano08",
    "10.226.47.85": "c2sragx04",
}

# Client Devices Section
st.markdown('<div class="section-header">Edge Devices</div>', unsafe_allow_html=True)

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
            st.caption(f"Device: **{AVAILABLE_DEVICES[selected_ip]}**")
            st.caption(f"IP: **{selected_ip}**")
            end_id = st.number_input("Last Client ID", min_value=1, value=1)
        
        add_btn = st.form_submit_button("Add Device")
        if add_btn:
            # Check if device already added
            if any(d['ip'] == selected_ip for d in st.session_state.clients):
                st.error(f"Device {AVAILABLE_DEVICES[selected_ip]} is already added")
            else:
                st.session_state.clients.append({
                    "name": AVAILABLE_DEVICES[selected_ip],
                    "ip": selected_ip,
                    "start": int(start_id),
                    "end": int(end_id)
                })
                st.rerun()

# Display devices
if st.session_state.clients:
    st.write(f"**{len(st.session_state.clients)} device(s) configured**")
    
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
                            st.success(f"Cleaned up containers on {device['name']}")
                        except subprocess.CalledProcessError as e:
                            st.error(f"Failed to cleanup {device['name']}")
        
        st.markdown("---")
else:
    st.info("No devices added yet")

# Topology Visualization
if st.session_state.clients:
    st.markdown('<div class="section-header">Network Topology</div>', unsafe_allow_html=True)
    
    dot = Digraph(format="png")
    dot.attr(rankdir='TB', bgcolor='transparent')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', color='#666666', penwidth='1.5')
    
    # Server
    dot.node('server', 
             f'{server_ip}:{server_port}\\nServer',
             shape='box',
             style='filled',
             fillcolor='#e8f4f8',
             color='#0066cc',
             penwidth='2')
    
    # Clients
    for idx, device in enumerate(st.session_state.clients):
        node_id = f'device_{idx}'
        num_clients = device['end'] - device['start'] + 1
        
        dot.node(node_id,
                f"{device['name']}\\n{device['ip']}\\n{num_clients} client(s)",
                shape='box',
                style='filled,rounded',
                fillcolor='#f0f0f0',
                color='#333333')
        
        dot.edge('server', node_id, dir='both', arrowhead='normal', arrowtail='normal')
    
    st.graphviz_chart(dot)

st.markdown("---")

# Deployment
st.markdown('<div class="section-header">Deployment</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.session_state.clients:
        if st.button("Deploy to All Devices", type="primary", use_container_width=True):
            for device in st.session_state.clients:
                with st.expander(f"Deploying to {device['name']}", expanded=True):
                    
                    # Build deployment script
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
                        st.success(f"✓ Deployed {device['end'] - device['start'] + 1} clients")
                    except subprocess.CalledProcessError as e:
                        st.error(f"✗ Deployment failed")
                        with st.expander("Error details"):
                            st.code(e.stderr if e.stderr else "Unknown error")
    else:
        st.warning("Add at least one device to begin deployment")

with col2:
    if st.button("Delete All Containers", type="secondary", use_container_width=True):
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
                except subprocess.CalledProcessError as e:
                    st.error(f"✗ Cleanup failed (device may be unreachable)")
                    with st.expander("Error details"):
                        st.code(e.stderr if e.stderr else "Unknown error")