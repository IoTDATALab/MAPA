import os
from params import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
from sympy import *
from math import sqrt
import math
import ComputePrivacy as Privacy

EPOCH = int(os.environ.get('EPOCH'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
RESULT_ROOT = os.environ.get('RESULT_ROOT')
gamma=0.2
c=torch.tensor(0.1)
lr=0.02
z=0.5
G = torch.tensor([0.01])

b=BATCH_SIZE
m=93024
T=1800
K_ASyn=3
delta=1./(m**1.1)
msgQueue = queue.Queue()

def list2array(grads):
    grads_ = []
    for i in range(len(grads)):
        grads_.extend(list(grads[i].numpy().ravel()))
    grads_ = np.array(grads_)
    return grads_
def zerosgrad(Layers_shape):
    zerograd = []
    for i in range(len(Layers_shape)):
        zerograd.append(torch.zeros(Layers_shape[i]))
    return zerograd

def Clipb(Clipbound, Layers_nodes, style="Flat"):
    Layer_nodes = torch.tensor(Layers_nodes).float()
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

def clip_update(Clipbound,Indicator, style="Geo"):
    std = z * torch.sqrt(1 / c) / K_ASyn
    Indicator = Indicator + std * torch.randn_like(Indicator)
    if style == "Lin":
        Clipbound = Clipbound - lr * (Indicator - gamma)
    if style == "Geo":
        Clipbound = Clipbound * torch.exp(-lr * (Indicator - gamma))
    return Clipbound

def Addnoise(Tensor, Clipbound):

    theta = zerosgrad(Layers_shape)
    std =  z * Clipbound.float().norm() *torch.sqrt(1/(1-c))/K_ASyn
    for i in range(len(Tensor)):
        theta[i] = Tensor[i] + std*torch.randn(Tensor[i].shape)
    return theta

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))
#def on_subscribe(client, userdata, mid, granted_qos):
#    print('subscribe successful')
def on_publish(client, userdata, mid):
    print('publish success')
def on_message(mqttc, obj, msg):
    print("received: " + msg.topic + " " + str(msg.qos))
    msglist = []
    msglist.append(msg.topic)
    msglist.append(msg.payload)
    msgQueue.put(msglist)

client = mqtt.Client()
client.on_connect = on_connect
#client.on_subscribe = on_subscribe
client.on_publish = on_publish
client.on_message = on_message
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe("ada2_grads/#", 2)
client.loop_start()

if __name__ == '__main__':
    epsilon=[]
    theta0 = InitializeParameters()
    Layers_nodes=[]
    Layers_shape=[]
    for i in range(len(theta0)):
        Layers_nodes.append(theta0[i].numel())
        Layers_shape.append(theta0[i].shape)
    Clipbound = Clipb(G, Layers_nodes, style="Flat")
    payload = [Clipbound, theta0]
    client.publish("init", cPickle.dumps(payload), 2)
    
    for epoch in range(EPOCH):
        for step in range(T + 1):
            print("step: ", step, "z:",z,)
            delta_sum = zerosgrad(Layers_shape)
            beta_sum = torch.zeros_like(Clipbound)
            edge_topic = []
            for i in range(K_ASyn):
                msglist = msgQueue.get()
                edgetopic = msglist[0]
                edgemsg = msglist[1]
                edgetopic = "ada2_params/" + edgetopic.split('/')[1]
                edge_topic.append(edgetopic)
                message_ = cPickle.loads(edgemsg)
                delta_ =message_[0]
                beta = message_[1]
                for i in range (len(delta_)):
                    delta_sum[i]+=delta_[i]
                for i in range(len(beta_sum)):
                    beta_sum[i]+=beta[i]
            for i in range (len(delta_sum)):
                 delta_sum[i] = delta_sum[i]/K_ASyn
            for i in range(len(beta_sum)):
                 beta_sum[i] = beta_sum[i]/K_ASyn
            grads=Addnoise(delta_sum, Clipbound)
            for i in range(len(grads)):
                theta0[i] = theta0[i].float()
                theta0[i] =theta0[i] + grads[i]

            Clipbound = clip_update(Clipbound, beta_sum, style="Geo")
            payload = [Clipbound, theta0]
            for i in range(len(edge_topic)):
                client.publish(edge_topic[i], cPickle.dumps(payload), 2)

            if step % 50 == 0:
                man_file = open(RESULT_ROOT + '[Adaclip2_Budget]', 'w')
                varepsilon = Privacy.ComputePrivacy(b / m, z, step+1, delta, 32)
                epsilon.append(varepsilon)
                print(epsilon, file=man_file)
                man_file.close()

            if epoch * (m / BATCH_SIZE) + step == T:
                 break



