import os
from params import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
import math
import ComputePrivacy as Privacy
import logging
logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)



MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
TEST_NUM = int(os.environ.get('TEST_NUM'))
RESULT_ROOT = os.environ.get('RESULT_ROOT')





z = 0.8
LR = 0.001
Clipbound = torch.tensor([0.001])
c = 0.1
m = 72377
T = 3000
BATCH_SIZE = 5

def init_deta(Layers_shape):
    deta = []
    for i in range(len(Layers_shape)):
        deta.append(torch.zeros(Layers_shape[i]))
    return deta


def ParamsUpdate(params, Deta, Clipbound):
    std = z * Clipbound.norm() * math.sqrt(1 / (1 - c)) ###flat
    for i in range(len(params)):
        #std = z * Clipbound[i].norm() * math.sqrt(1 / (1 - c)) 
        params[i] = params[i].float() + Deta[i] + std * torch.randn(Deta[i].shape)
    return params


def ClipUpdate(Clipbound, Indicator, style="Geo"):
    std = z * math.sqrt(1 / c)  
    Indicator = Indicator + std * torch.randn_like(Indicator)
    if style == "Lin":
        Clipbound = Clipbound - LR * (Indicator - 0.5)
    if style == "Geo":
        Clipbound = Clipbound * torch.exp(-LR * (Indicator - 0.5))
    return Clipbound


msgQueue = queue.Queue()


def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))


def on_message(mqttc, obj, msg):
    print("received: " + msg.topic + " " + str(msg.qos))
    msglist = []
    msglist.append(msg.topic)
    msglist.append(msg.payload)
    msgQueue.put(msglist)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
#client.enable_logger(logger)
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe("adaclip2/#", 2)
client.loop_start()

if __name__ == '__main__':
 
    epsilon = []
    Sampled_ratio = BATCH_SIZE / m
    delta = 1. / m**1.1

    params = InitializeParameters()
    payload = [Clipbound, params]
    client.publish("init_bsl", cPickle.dumps(payload), 2)

    Layers_shape = []
    for i in range(len(params)):
        Layers_shape.append(params[i].shape)
    
    
    for t in range(T):
        msglist = msgQueue.get()
        edgetopic = msglist[0]
        edgemsg = msglist[1]
        edgetopic = "adaclip2_params/" + edgetopic.split('/')[1]
        mesg = cPickle.loads(edgemsg)
        deta = mesg[0]
        Indicator = mesg[1].type(torch.FloatTensor)
        Clipbound = mesg[2]

        params = ParamsUpdate(params, deta, Clipbound)
        Clipbound = ClipUpdate(Clipbound, Indicator, style="Geo")


        payload = [Clipbound, params]

        client.publish(edgetopic, cPickle.dumps(payload), 2)
        
        if t % TEST_NUM == 0:
            man_file = open(RESULT_ROOT + '[Adaclip2_Budget]', 'w')
            varepsilon = Privacy.ComputePrivacy(Sampled_ratio, z, t+1, delta, 32)
            epsilon.append(varepsilon)
            print(epsilon, file=man_file)
            man_file.close()

    print(T)
