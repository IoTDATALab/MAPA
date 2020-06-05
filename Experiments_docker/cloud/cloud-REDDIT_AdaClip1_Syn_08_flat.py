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
TEST_NUM = int(os.environ.get('TEST_NUM'))
RESULT_ROOT = os.environ.get('RESULT_ROOT')
DELAY = int(os.environ.get('DELAY'))


LR = 3.5
z = 0.8
m = 72377
T = 1000
BATCH_SIZE = 5

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
#client.enable_logger(logger)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe([("adaclip1_grads/#", 2)])
client.loop_start()

if __name__ == '__main__':

    params = InitializeParameters()
    client.publish("init", cPickle.dumps(params), 2)

    epsilon = []
    Sampled_ratio = DELAY * BATCH_SIZE / m
    delta = 1. / m ** 1.1

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
        msg = [params, grads_sum]
        client.publish("adaclip1_params", cPickle.dumps(msg), 2)


        if t % TEST_NUM == 0:
            man_file = open(RESULT_ROOT + '[Adaclip1_Budget]', 'w')
            varepsilon = Privacy.ComputePrivacy(Sampled_ratio, z, t + 1, delta, 32)
            epsilon.append(varepsilon)
            print(epsilon, file=man_file)
            man_file.close()

    print(T)
