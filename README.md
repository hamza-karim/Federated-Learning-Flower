# Federated Learning for Anomaly Detection in Distributed Energy Resources (DERs)
## Fed learning project with flwr client and server


This repository contains the code, datasets, and documentation for our research paper titled "Integrating Edge Computing and Federated Learning for Enhanced Anomaly Detection in DERs." This study explores the application of Federated Learning (FL) in a distributed energy resource (DER) environment, with a focus on anomaly detection using NVIDIA Jetson devices.

The repository demonstrates Federated Learning-based anomaly detection in a DER environment using the Flower framework. The project specifically employs Long Short-Term Memory (LSTM) autoencoders to detect anomalies related to False Data Injection Attacks (FDIAs). The deployment is managed using Docker containers, and pre-built images can be found at:

https://hub.docker.com/repositories/hamzakarim07
1. **hamzakarim07 / flwr_client**
2. **hamzakarim07 / flwr server**

<div align="center">
  <img src="images/fed framework.png" alt="fed framework" width="300">
</div>

### How to run:
**Prerequisites:**
1. NVIDIA Jetson devices (e.g., Jetson AGX Orin, Jetson Nano)
2. Docker installed on the devices

**Setup:**
1. Pull Docker image for Server and Client from : ("https://hub.docker.com/repositories/hamzakarim07").
2. 
Client:
sudo docker-compose up -d 
python client_rf.py --ip=172.16.232.50 --folder=Client_1 --id=1

Now i have to implement LSTM autoencoders ( this is repo testing )


server: python server_LSTM_2.py

client1: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_1_LSTM --id=1

client2: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_22_LSTM --id=2

sudo docker build -t hamzakarim07/flwr_client:latest -f FL_client/docker/Dockerfile .

sudo docker build -t hamzakarim07/flwr_server:latest -f FL_server/docker/Dockerfile .