import os
from params import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
import math

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
DELAY = int(os.environ.get('DELAY'))


LR = 0.1
T = 1000
msgQueue = queue.Queue()
def Init_Gradient(params):
    grads = []
    for i in range(len(params)):
        grads.append(torch.zeros(params[i].shape))
    return grads

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))


def on_message(mqttc, obj, msg):
    print("received: " + msg.topic + " " + str(msg.qos))
    msgQueue.put(msg.payload)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
#client.enable_logger(logger)
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe("nondp_grads/#", 2)
client.loop_start()

if __name__ == '__main__':

    params = InitializeParameters()
    client.publish("init", cPickle.dumps(params), 2)

    for t in range(T):

        grads_sum = Init_Gradient(params)
        for i in range(DELAY):
            grads = cPickle.loads(msgQueue.get())
            for i in range(len(grads)):
                grads_sum[i] += grads[i]

        #avg grads
        for i in range(len(grads_sum)):
            grads_sum[i] /= DELAY

        for i in range(len(params)):
            params[i] = params[i].float()
            params[i] -= LR * grads_sum[i]

        client.publish("nondp_params", cPickle.dumps(params), 2)

    print(T)
