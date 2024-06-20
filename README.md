# Federated-Learning-
Fed learning project with (flwr) client and server.
this is in testing stage right now


Client:
sudo docker-compose up -d 
python client_rf.py --ip=172.16.232.50 --folder=Client_1 --id=1

Now i have to implement LSTM autoencoders ( this is repo testing )


server: python server_LSTM_2.py
client2: python client_LSTM_2.py --ip=192.168.1.23 --folder=Client_22_LSTM --id=2