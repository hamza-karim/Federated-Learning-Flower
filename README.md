# Federated-Learning-Flower
**Fed learning project with flwr client and server.**

This repo demonstrate the Federated Learning in a DER (Distributed Energy Resource) using Flower. 
Project is deployed using docker containers and pre-built images can be found at: https://hub.docker.com/repositories/hamzakarim07
1. **hamzakarim07 / flwr_client**
2. **hamzakarim07 / flwr server**

<div align="center">
  <img src="images/fed framework.png" alt="fed framework" width="300">
</div>


Client:
sudo docker-compose up -d 
python client_rf.py --ip=172.16.232.50 --folder=Client_1 --id=1

Now i have to implement LSTM autoencoders ( this is repo testing )


server: python server_LSTM_2.py

client1: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_1_LSTM --id=1

client2: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_22_LSTM --id=2

sudo docker build -t hamzakarim07/flwr_client:latest -f FL_client/docker/Dockerfile .

sudo docker build -t hamzakarim07/flwr_server:latest -f FL_server/docker/Dockerfile .