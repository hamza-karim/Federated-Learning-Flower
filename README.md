# Federated-Learning-Flower
Fed learning project with flwr client and server.
![fed framework.png](..%2F..%2Fwork%2Fpaper%20diagrams%2Ffed%20framework.png)


Client:
sudo docker-compose up -d 
python client_rf.py --ip=172.16.232.50 --folder=Client_1 --id=1

Now i have to implement LSTM autoencoders ( this is repo testing )


server: python server_LSTM_2.py

client1: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_1_LSTM --id=1

client2: python3 client_LSTM_2.py --ip=192.168.1.23 --folder=Client_22_LSTM --id=2

sudo docker build -t hamzakarim07/flwr_client:latest -f FL_client/docker/Dockerfile .

sudo docker build -t hamzakarim07/flwr_server:latest -f FL_server/docker/Dockerfile .