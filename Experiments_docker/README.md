#  Asynchronous Federated Learning with Differential Privacy for Edge Intelligence
Multi-stage adjustable private algorithm(MAPA) by computational differential privacy (DP) to protect the training process,focusing on the impacts of noise on the machine learning in AFL and a better trade-off between the model accuracy and privacy guarantee to improve the trade-off by dynamically adjusting the variance of noise

In order to compare the algorithm,we set up seven comparison methods for synchronous and asynchronous mode,including NonDP,FixDP-S,FixDP-C,MAPA-S,MAPA-C,AdaClip1 and AdaClip2. And the algorithm is encapsulated in the docker container and defined by the makefile.
## 1.Environment Deployment

|     Python     |     3.6      |
|     Docker     |   18.09.7    |
| Docker-compose |    1.24.1    |
|    OpenSSH     |     7.6      |
## 2.Network Configuration
Before the application runs, first perform the network configuration of the user operating device and the working node, and write the network topology to the `ssh_config` file, such asï¼?
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

## 3.Parameter settings
Algorithm-related parameters are defined in the Makefileï¼Œsuch asï¼?
```
# In cloud 
CLOUD := cloud                
MQTT_IP ?= xxx.xxx.xxx.xxx        #cloud device ip
MQTT_PORT ?= 1884               #cloud device MQTT broker port

# In edge  
EDGES := edge1                 #edge device name
EDGES_NUM ?=1                   #edge device numbers
METHOD ?= NonDP                #NonDP,FixDP-S,FixDP-C,MAPA-S,MAPA-C,AdaClip1,AdaClip2
BATCH_SIZE ?= 5
EPOCH ?= 1
TEST_NUM ?= 100               #Test every TEST_NUM iterations
RESULT_ROOT ?= './result/'             #result directory  

```

## 4.Run
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


