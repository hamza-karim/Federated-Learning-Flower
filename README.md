# Federated Learning for Anomaly Detection in Distributed Energy Resources (DERs)

## Fed Learning Project with Flower (flwr) Client and Server

This repository contains the code, datasets, and documentation for our research paper titled **"Integrating Edge Computing and Federated Learning for Enhanced Anomaly Detection in DERs."** This study explores the application of Federated Learning (FL) in a distributed energy resource (DER) environment, with a focus on anomaly detection using NVIDIA Jetson devices.

The repository demonstrates Federated Learning-based anomaly detection in a DER environment using the Flower framework. The project specifically employs Long Short-Term Memory (LSTM) autoencoders to detect anomalies related to False Data Injection Attacks (FDIAs). The deployment is managed using Docker containers, and pre-built images can be found at:

- [hamzakarim07/flwr_client](https://hub.docker.com/repositories/hamzakarim07)
- [hamzakarim07/flwr_server](https://hub.docker.com/repositories/hamzakarim07)

<div align="center">
  <img src="images/fed_framework.png" alt="Federated Learning Framework" width="1000">
</div>

### How to Run:

#### Prerequisites:
1. NVIDIA Jetson devices (e.g., Jetson AGX Orin, Jetson Nano)
2. Docker installed on the devices

#### Setup Federated Server:
1. Pull the Docker image for the client on the Jetson Nano device from [Docker Hub](https://hub.docker.com/repositories/hamzakarim07):
   ```bash
   sudo docker pull hamzakarim07/flwr_server:latest
2. Run following command to run docker container:
   ```bash
   sudo docker run -d --name flwr-server --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all -p 8080:8080 hamzakarim07/flwr_server:latest
3. Go inside the docker container:
   ```bash
   sudo docker exec -it <container_id_or_name> bash
   cd src
4. Run python command to execute the python script:
   ```bash
   python3 server_LSTM_2.py
   
#### Setup Federated Clients:
1. Pull the Docker image for the client on the Jetson Nano device from [Docker Hub](https://hub.docker.com/repositories/hamzakarim07):
   ```bash
   sudo docker pull hamzakarim07/flwr_client:latest
2. Run following command to run docker container for client 1, just change the name of container for client 2:
   ```bash
   sudo docker run -d --name flwr-client1 --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all hamzakarim07/flwr_client:latest
3. Go inside the docker container:
   ```bash
   sudo docker exec -it <container_id_or_name> bash
   cd src
4. Run python command to execute the python script for client 1 and 2 **--ip is server ip**:
   ```bash
   python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_1_LSTM --id=1
   python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_22_LSTM --id=2
   
Client:
sudo docker-compose up -d 
python client_rf.py --ip=172.16.232.50 --folder=Client_1 --id=1




server: python3 server_LSTM_2.py

client1: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_1_LSTM --id=1

client2: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_22_LSTM --id=2

sudo docker build -t hamzakarim07/flwr_client:latest -f FL_client/docker/Dockerfile .

sudo docker build -t hamzakarim07/flwr_server:latest -f FL_server/docker/Dockerfile .



sudo docker push hamzakarim07/flwr_server:latest


sudo nano /root/.docker/config.json


if docker is removed:
sudo apt-get remove --purge docker docker.io containerd runc
sudo apt-get update
sudo apt-get install docker.io

sudo systemctl start docker
sudo systemctl enable docker
sudo docker run hello-world

check docker settings for nvidia container runtime:
sudo nano /etc/docker/daemon.json
