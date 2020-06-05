import os
from params import *
import _pickle as cPickle
import paho.mqtt.client as mqtt
import queue
from math import sqrt
import math
import ComputePrivacy as Privacy

EPOCH = int(os.environ.get('EPOCH'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
RESULT_ROOT = os.environ.get('RESULT_ROOT')

K_ASyn = 3
LR = 0.8
m = 93024
z = 0.5

def init_grads(params):
    grads = []
    for i in range(len(params)):
        grads.append(torch.zeros(params[i].shape))

    return grads

msgQueue = queue.Queue()
def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))
def on_publish(client, userdata, mid):
    print('publish success')
def on_message(mqttc, obj, msg):
    print("received: " + msg.topic + " " + str(msg.qos))
    msglist = []
    msglist.append(msg.topic)
    msglist.append(msg.payload)
    msgQueue.put(msglist)

client = mqtt.Client()
client.on_connect=on_connect
client.on_publish=on_publish
client.on_message = on_message
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe([("ada1_grads/#", 2)])
client.loop_start()

if __name__ == '__main__':
    epsilon=[]
    T = 1800
    b=BATCH_SIZE
    delta = 1. / (m ** 1.1)
    params = InitializeParameters()
    client.publish("init", cPickle.dumps(params), 2)
    for epoch in range(EPOCH):
       for step in range(T + 1):
          print("step: ",step)
          edge_topic = []
          grads_sum = init_grads(params)
          for i in range(K_ASyn):
              msglist = msgQueue.get()
              edgetopic = msglist[0]
              edgemsg = msglist[1]
              edgetopic = "ada1_params/" + edgetopic.split('/')[1]
              edge_topic.append(edgetopic)
              grads = cPickle.loads(edgemsg)
              for i in range(len(grads)):
                  grads_sum[i] += grads[i]
          for i in range(len(grads_sum)):
              grads_sum[i] /= K_ASyn

          for i in range(len(params)):
              params[i] = params[i].float()
              params[i] -= LR * grads_sum[i]

          payload=[grads_sum,params]
          for i in range(len(edge_topic)):
              client.publish(edge_topic[i], cPickle.dumps(payload), 2)
              
          if step % 50==0:
              man_file = open(RESULT_ROOT + '[Adaclip1_Budget]', 'w')
              varepsilon = Privacy.ComputePrivacy(b / m, z, step+1, delta, 32)
              epsilon.append(varepsilon)
              print(epsilon, file=man_file)
              man_file.close()

          if epoch * (m / BATCH_SIZE) + step == T:
              break














