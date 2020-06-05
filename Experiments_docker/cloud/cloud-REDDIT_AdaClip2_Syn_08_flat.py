import os
from params import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
import math
import ComputePrivacy as Privacy
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
DELAY = int(os.environ.get('DELAY'))
TEST_NUM = int(os.environ.get('TEST_NUM'))
RESULT_ROOT = os.environ.get('RESULT_ROOT')


Clipbound = 0.01
z = 0.8
c = 0.1
LR = 0.01
m = 72377
T = 1000
BATCH_SIZE = 5

def init_deta(Layers_shape):
    deta = []
    for i in range(len(Layers_shape)):
        deta.append(torch.zeros(Layers_shape[i]))
    return deta

def ParamsUpdate(params, Deta, Clipbound):
    std = z * Clipbound.norm() * math.sqrt(1 / (1 - c)) / DELAY
    for i in range(len(params)):
        #std = z * Clipbound.norm() * math.sqrt(1 / (1 - c)) / DELAY
        params[i] = params[i].float() + Deta[i] + std * torch.randn(params[i].shape)
    return params


def ClipUpdate(Clipbound, Indicator_, style="Geo"):
    std = z * math.sqrt(1 / c) / DELAY
    Indicator = Indicator_ + std * torch.randn_like(Indicator_)
    if style == "Lin":
        Clipbound = Clipbound - LR * (Indicator - 0.5)
    if style == "Geo":
        Clipbound = Clipbound * torch.exp(-LR * (Indicator - 0.5))
    return Clipbound


msgQueue = queue.Queue()


def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))


def on_message(mqttc, obj, msg):
    # print("received: " + msg.topic + " " + str(msg.qos))
    msgQueue.put(msg.payload)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
#client.enable_logger(logger)
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe("adaclip2/#", 2)
client.loop_start()

if __name__ == '__main__':

    epsilon = []
    Sampled_ratio = DELAY * BATCH_SIZE / m
    delta = 1. / m ** 1.1

    params = InitializeParameters()
    payload = [Clipbound, params]
    client.publish("init", cPickle.dumps(payload), 2)

    Layers_shape = []
    for i in range(len(params)):
        Layers_shape.append(params[i].shape)

    for t in range(T):

        Indicator_ = torch.tensor([0.])    ##############
        Deta = init_deta(Layers_shape)
        for i in range(DELAY):
            msg = cPickle.loads(msgQueue.get())
            deta = msg[0]
            Indicator = msg[1].type(torch.FloatTensor)
            Clipbound = msg[2]
            for i in range(len(deta)):
                Deta[i] += deta[i]
            Indicator_ += Indicator

        for i in range(len(Deta)):
            Deta[i] /= DELAY
        Indicator_ /= DELAY
        print(Clipbound)
        params = ParamsUpdate(params, Deta, Clipbound)
        Clipbound = ClipUpdate(Clipbound, Indicator_, style="Geo")

        payload = [Clipbound, params]

        client.publish("adaclip2_params", cPickle.dumps(payload), 2)

        if t % TEST_NUM == 0:
            man_file = open(RESULT_ROOT + '[Adaclip2_Budget]', 'w')
            varepsilon = Privacy.ComputePrivacy(Sampled_ratio, z, t + 1, delta, 32)
            epsilon.append(varepsilon)
            print(epsilon, file=man_file)
            man_file.close()

    print(T)

