import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import paho.mqtt.client as mqtt
import _pickle as cPickle
from math import sqrt
import numpy as np
import os, queue, random, time
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
import math
from sympy import *

CLIENT_ID = str(random.random())
EPOCH = int(os.environ.get('EPOCH'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
TEST_NUM = int(os.environ.get('TEST_NUM'))
NUM = int(os.environ.get('NUM'))
EDGE_NAME=os.environ.get('EDGE_NAME')
RESULT_ROOT = os.environ.get('RESULT_ROOT')

class Femnist(Dataset):
    def __init__(self, data_path, train=True, transform=None):

        self.train = train
        self.transform = transform

        if self.train:
            self.train_x = []
            self.train_y = []
            for number in range(NUM,NUM+2):
                with open(data_path + 'all_data_' + str(number) + '_niid_0_keep_0_train_9.json', 'r') as f:
                    train = json.load(f)
                    for u in train['users']:
                        self.train_x += train['user_data'][u]['x']
                        self.train_y += train['user_data'][u]['y']

        else:
            self.test_x = []
            self.test_y = []
            for number in range(NUM,NUM+2):
                with open(data_path + 'all_data_' + str(number) + '_niid_0_keep_0_test_9.json', 'r') as f:
                    test = json.load(f)
                    for u in test['users']:
                        self.test_x += test['user_data'][u]['x']
                        self.test_y += test['user_data'][u]['y']

    def reshape_input(self, array):
        return np.asarray(array).reshape((28, 28, 1))

    def __len__(self):
        if self.train:
            return len(self.train_y)
        else:
            return len(self.test_y)

    def __getitem__(self, idx):
        if self.train:
            x = self.train_x[idx]
            y = self.train_y[idx]
            img, label = self.reshape_input(x), np.array(y)
            img = self.transform(img).type(torch.FloatTensor)
            label = torch.tensor(label).type(torch.LongTensor)
            return img, label
        else:
            x = self.test_x[idx]
            y = self.test_y[idx]
            img, label = self.reshape_input(x), np.array(y)
            img = self.transform(img).type(torch.FloatTensor)
            label = torch.tensor(label).type(torch.LongTensor)
            return img, label


train_data = Femnist(
    './data/femnist/train/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
)
test_data = Femnist(
    './data/femnist/test/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

def zerosgrad(Layers_shape):
    zerograd = []
    for i in range(len(Layers_shape)):
        zerograd.append(torch.zeros(Layers_shape[i]))
    return zerograd

def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())

    return Layers_num+1, Layers_shape, Layers_nodes

def Clipb(Clipbound, Layers_nodes, style="Flat"):
    Layer_nodes = torch.tensor(Layers_nodes).float()
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

def Clip(Tensor, ClipBound):
    norm = 0
    Layers_num = len(Tensor)
    Tuple_num = torch.ones(Layers_num)
    for i in range(Layers_num):
        norm = norm + Tensor[i].float().norm()**2
        Tuple_num[i] = Tensor[i].numel()
    norm = norm.sqrt()
    if len(ClipBound) == 1:
        Indicator = 1*(norm<=ClipBound[0])
        for i in range(Layers_num):
            Tensor[i] = Tensor[i]*torch.min(torch.ones(1), ClipBound[0]/norm)
    if len(ClipBound) > 1:
        Indicator = torch.zeros(Layers_num)
        # ClipBound_layers = Tuple_num.float()/Tuple_num.norm()*ClipBound#
        for i in range(Layers_num):
            norm_layer = Tensor[i].float().norm()
            Indicator[i] = 1*(norm_layer<=ClipBound[i])
            Tensor[i] = Tensor[i]*torch.min(torch.ones(1), ClipBound[i]/norm_layer)
    return Tensor, Indicator

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(
                in_channels=1,              
                out_channels=10,            
                kernel_size=5,                            
                  ),                              
            nn.ReLU(),                     
            nn.MaxPool2d(2),    
        )
        self.conv2 = nn.Sequential(        
            nn.Conv2d(
                in_channels=10,             
                out_channels=10,          
                kernel_size=5,  
                     
                ),    
            nn.ReLU(),                     
            nn.MaxPool2d(2),              
        )
        self.out = nn.Linear(10 * 4 * 4, 62)
         
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)         
        output = self.out(x)
        return output, x
cnn=CNN()
msgQueue = queue.Queue()
def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))
def on_subscribe(client, userdata, mid, granted_qos): 
    print('subscribe successful')
def on_publish(client, userdata, mid):
    print('publish success')
def on_message(mqttc, obj, msg):
    #print("received: " + msg.topic + " " + str(msg.qos))
    msgQueue.put(msg.payload)
client = mqtt.Client(client_id = CLIENT_ID)
client.on_connect = on_connect
#client.on_subscribe = on_subscribe
#client.on_publish = on_publish
client.on_message = on_message
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe([("init", 2), ("mapa_params/" + CLIENT_ID, 2)])
client.loop_start()

if __name__=='__main__':
    acc=[]
    Losstest=[]
    Losstrain=[]
    Layers_num, Layers_shape, Layers_nodes = GetModelLayers(cnn)
    p = cPickle.loads(msgQueue.get())
    z=p[0]
    G=p[1]
    params =p[2]
    for i,f in enumerate(cnn.parameters()):
        f.data = params[i].float()
    for epoch in range(EPOCH):
        train_loss = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn.forward(b_x)[0]
            loss = nn.CrossEntropyLoss()(output, b_y)
            cnn.zero_grad()
            loss.backward()
            train_loss += loss.item()
            if step % TEST_NUM == 0:
                total = 0
                correct = 0
                test_loss = 0
                for i, (test_x, test_y) in enumerate(test_loader):
                    test_output, last_layer = cnn(test_x)
                    test_loss += nn.CrossEntropyLoss()(test_output, test_y).item()
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    total += float(test_y.size(0))
                    correct += float((pred_y == test_y.data.numpy()).astype(int).sum())

                man_file1 = open(RESULT_ROOT + '[' + str(EDGE_NAME) + ']' + '[MAPA-Accuracy]', 'w')
                accuracy = correct / total
                acc.append(accuracy)
                print(acc, file=man_file1)
                man_file1.close()

                man_file2 = open(RESULT_ROOT + '[' + str(EDGE_NAME) + ']' + '[MAPA-TestLoss]', 'w')
                test_loss = test_loss / (i + 1)
                Losstest.append(test_loss)
                print(Losstest, file=man_file2)
                man_file2.close()

                man_file3 = open(RESULT_ROOT + '[' + str(EDGE_NAME) + ']' + '[MAPA-TrainLoss]', 'w')
                Losstrain.append(train_loss / (step + 1))
                print(Losstrain, file=man_file3)
                man_file3.close()
                
                print("step:",step,"acc:",accuracy,"loss: ",test_loss,"train_loss: ",train_loss/(step+1))
            w_c1 = cnn.conv1[0].weight.grad
            b_c1 = cnn.conv1[0].bias.grad
            w_c2 = cnn.conv2[0].weight.grad
            b_c2 = cnn.conv2[0].bias.grad
            w_o = cnn.out.weight.grad
            b_o = cnn.out.bias.grad
            grads = [w_c1, b_c1, w_c2, b_c2, w_o, b_o]
            ClipBound = Clipb(G, Layers_nodes, style="Flat")
            grads_ = Clip(grads, ClipBound)[0]
            payload=[grads_,ClipBound]
            client.publish("mapa_grads/" + CLIENT_ID, cPickle.dumps(payload), 2)  #
            p = cPickle.loads(msgQueue.get())
            z = p[0]
            G = p[1]
            params = p[2]
            for i, f in enumerate(cnn.parameters()):
                f.data = params[i].float()

 




