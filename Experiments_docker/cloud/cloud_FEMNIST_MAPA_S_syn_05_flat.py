import os
from params import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
from sympy import *
from math import sqrt
import math
import random
import ComputePrivacy as Privacy
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

EPOCH = int(os.environ.get('EPOCH'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
RESULT_ROOT = os.environ.get('RESULT_ROOT')
b=BATCH_SIZE
msgQueue = queue.Queue()

def init_grads(params):
    grads = []
    for i in range(len(params)):
        grads.append(torch.zeros(params[i].shape))
    return grads

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))
def on_subscribe(client, userdata, mid, granted_qos): 
    print('subscribe successful')
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
client.on_subscribe = on_subscribe
client.on_publish = on_publish
client.on_message = on_message
client.connect(MQTT_IP,MQTT_PORT, 600)
client.subscribe("mapa_grads/#", 2)
client.loop_start()

if __name__=='__main__':
    epsilon=[]
    Layers_nodes = []
    params = InitializeParameters()
    for i in range(len(params)):
        Layers_nodes.append(params[i].numel())
    K_ASyn = 3
    m = 93024
    delta = 1 / (m ** 1.1)
    z = 0.5
    T = 1800
    G = torch.tensor([15.])
    SIGMA = 4
    L = 0.0185
    theta = torch.tensor([0.8])
    R = 50
    tau = torch.tensor([0.])
    Delta_b = (SIGMA**2 + sum(Layers_nodes) * (z * G)**2) / b
    P = 2 * Delta_b / (theta**2 * G**2 * (tau + 1))
    LR = 1/(2*P*L*(tau+1))
    T0 = torch.ceil(4*P**2*L*(tau+1)**2 * R / Delta_b)
    payload = [z,G,params]
    client.publish("init", cPickle.dumps(payload), 2)

    for epoch in range(EPOCH):
          for step in range(T+1):
                print("step:",step,"LR:",LR,"G:",G,"T0:",T0)
                edge_topic = []
                grads_sum = init_grads(params)
                for i in range(K_ASyn):
                    msglist = msgQueue.get()
                    edgetopic = msglist[0]
                    edgemsg = msglist[1]
                    edgetopic = "mapa_params/" + edgetopic.split('/')[1]
                    edge_topic.append(edgetopic)
                    grads = cPickle.loads(edgemsg)
                    for i in range(len(grads)):
                        grads_sum[i] += grads[i]
                for i in range(len(grads_sum)):
                    grads_sum[i] /= K_ASyn

                for i in range(len(params)):
                    params[i] = params[i].float()
                    params[i] -= LR [0]* grads_sum[i]
                    
                payload = [z, G, params]
                for i in range(len(edge_topic)):
                    client.publish(edge_topic[i], cPickle.dumps(payload), 2)

                if step % 50==0:
                    man_file = open(RESULT_ROOT + '[MAPA_Budget]', 'w')
                    varepsilon = Privacy.ComputePrivacy(b / m, z, step+1, delta, 32)
                    epsilon.append(varepsilon)
                    print(epsilon, file=man_file)
                    man_file.close()

                if epoch*(m/BATCH_SIZE)+step >=T0[0]:  
                    G=G*theta
                    Delta_b = (SIGMA ** 2 + sum(Layers_nodes) * (z * G) ** 2) / b
                    P = 2 * Delta_b / (theta ** 2 * G ** 2 * (tau + 1))
                    LR = 1 / (2 * P * L * (tau + 1))
                    T0 = torch.ceil(4 * P ** 2 * L * (tau + 1) ** 2 * R / Delta_b)+T0
                    continue

                if epoch*(m/BATCH_SIZE)+step==T:
                    break

