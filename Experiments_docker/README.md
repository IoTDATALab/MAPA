#  Asynchronous Federated Learning with Differential Privacy for Edge Intelligence
Multi-stage adjustable private algorithm(MAPA) by computational differential privacy (DP) to protect the training process,focusing on the impacts of noise on the machine learning in AFL and a better trade-off between the model accuracy and privacy guarantee to improve the trade-off by dynamically adjusting the variance of noise

In order to compare the algorithm,we set up seven comparison methods for synchronous and asynchronous mode,including NonDP,FixDP-S,FixDP-C,MAPA-S,MAPA-C,AdaClip1 and AdaClip2. And the algorithm is encapsulated in the docker container and defined by the makefile.
## 1. Environment Deployment

|     Python     |     3.6      |
|     Docker     |   18.09.7    |
| Docker-compose |    1.24.1    |
|    OpenSSH     |     7.6      |

## 2. Project structure

```
|----cloud
|     |----cloud-REDDIT_NonDP_Asyn_08_flat.py       
|     |----cloud-REDDIT_FixDP_S_Asyn_08_flat.py            
|     |----cloud-REDDIT_FixDP_C_Asyn_08_flat.py  
|     |----cloud-REDDIT_MAPA_S_Asyn_08_flat.py  
         ......
|    ©À©¤©¤ cloud-FEMNIST_NonDP_Asyn_05_flat.py
|    ©À©¤©¤ cloud-FEMNIST_FixDP_S_Asyn_05_flat.py
|    ©À©¤©¤ cloud-FEMNIST_FixDP_C_Asyn_05_flat.py
|    ©À©¤©¤ cloud-FEMNIST_MAPA_S_Asyn_05_flat.py
         ......       
|    ©À©¤©¤ Dockerfile               Docker image build file
|    ©À©¤©¤ docker-compose.yml       Defining YAML files for services, networks, and volumes
|    ©À©¤©¤ docker-compose.mqtt.yml  Edge and cloud communication
|    ©À©¤©¤ ComputePrivacy.py        Calculate privacy budget
|    ©À©¤©¤ params.py                Initialization parameters of LSTM model
|    ©À©¤©¤ params_f.py              Initialization parameters of CNN model
|    ©¸©¤©¤  result                              Output
          ©¸©¤©¤ [METHOD_Budget].txt      

|
©À©¤©¤edge  
|    ©À©¤©¤ edge-REDDIT_NonDP_Asyn_08_flat.py       
|    ©À©¤©¤ edge-REDDIT_FixDP_S_Asyn_08_flat.py            
|    ©À©¤©¤ edge-REDDIT_FixDP_C_Asyn_08_flat.py  
|    ©À©¤©¤ edge-REDDIT_MAPA_S_Asyn_08_flat.py  
         ......
|    ©À©¤©¤ edge-FEMNIST_NonDP_Asyn_05_flat.py
|    ©À©¤©¤ edge-FEMNIST_FixDP_S_Asyn_05_flat.py
|    ©À©¤©¤ edge-FEMNIST_FixDP_C_Asyn_05_flat.py
|    ©À©¤©¤ edge-FEMNIST_MAPA_S_Asyn_05_flat.py
         ......
|    ©À©¤©¤ Dockerfile    
|    ©À©¤©¤ docker-compose.yml              
|    ©À©¤©¤ data                               Datasets
|    ©À   ©¸©¤©¤FEMNIST ¡¢REDDIT
|    ©¸©¤©¤  result                              Output
|         ©¸©¤©¤ [EDGE_NAME][METHOD-Accuracy].txt    
          ©¸©¤©¤ [EDGE_NAME][METHOD-TestLoss].txt    
          ©¸©¤©¤ [EDGE_NAME][METHOD-TrainLoss].txt  

©À©¤©¤Makefile                          Setting parameters    
©À©¤©¤README.md
©¸©¤©¤ssh_config

```

## 3. Network Configuration
Before the application runs, first perform the network configuration of the user operating device and the working node, and write the network topology to the `ssh_config` file, such as:
```
Host cloud
    HostName xxx.xxx.xxx.xxx
    Port 22
    User node2user

Host edge1                                     #Node name
    HostName xxx.xxx.xxx.xxx                   #ip
    Port 22                                    #port
    User node1user                             #ssh connection username

```

## 4. Parameter settings
Algorithm-related parameters are defined in the Makefile,such as:
```
# In cloud 
CLOUD := cloud                
MQTT_IP ?= xxx.xxx.xxx.xxx           #cloud device ip
MQTT_PORT ?= 1884                    #cloud device MQTT broker port

# In edge  
EDGES := edge1                       #edge device name
METHOD ?= REDDIT_MAPA_S_Asyn_08_flat #REDDIT_MAPA_C_Syn_08_flat FEMNIST_MAPA_C_Syn_05_flat...
TEST_NUM ?= 30                       #Test every TEST_NUM iterations
RESULT_ROOT ?= './result/'           #result directory  

```

## 5. Run
Before using our model, download the datasets from http://leaf.cmu.edu and put them under the data folder. 
Under the project folder, execute the shell command to apply the project:
* Perform network configuration between the user-operated device and the working node, including the configuration of ssh-free operation. After completing this step, the user operates the device to perform algorithm-related steps on other working nodes directly through ssh.
```
make net_config
```
* The program and algorithm data packets are transmitted from the user operating device to the corresponding working node and mirrored.
```
make build
```
* Start the container, transfer the parameters to each working node, assign tasks to the working nodes, and start the algorithm work.
```
make run
```
* Look at the work log on the master node, and the log content is dynamically updated in real time following the current program execution.
```
make logs
```
* Output the experimental results to the host.

```
make result
```
* Clean up all containers and free up resources.

```
make clean
```


