import os
from params import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
import math

import logging

logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')



LR = 0.01
T = 3000
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
client.subscribe("nondp_grads/#", 2)
client.loop_start()

if __name__ == '__main__':

   
    params = InitializeParameters()
    client.publish("init", cPickle.dumps(params), 2)

    for t in range(T):

        
        msglist = msgQueue.get()
        edgetopic = msglist[0]
        edgemsg = msglist[1]
        edgetopic = "nondp_params/" + edgetopic.split('/')[1]
        grads = cPickle.loads(edgemsg)

        for i in range(len(params)):
            params[i] = params[i].float()
            params[i] -= LR * grads[i]


        client.publish(edgetopic, cPickle.dumps(params), 2)

        
    client.publish("Halt", cPickle.dumps(params), 2)
    print(T)
