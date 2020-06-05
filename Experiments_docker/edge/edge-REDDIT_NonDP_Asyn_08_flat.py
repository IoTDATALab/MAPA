import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch.nn.functional as F
import paho.mqtt.client as mqtt
import _pickle as cPickle
from math import sqrt
import numpy as np
import os, queue, random
import matplotlib.pyplot as plt
import json
import pickle
import collections

#import logging
#logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger(__name__)

CLIENT_ID = str(random.random())
EPOCH = int(os.environ.get('EPOCH'))
MQTT_PORT = int(os.environ.get('MQTT_PORT'))
MQTT_IP = os.environ.get('MQTT_IP')
TEST_NUM = int(os.environ.get('TEST_NUM'))
RESULT_ROOT = os.environ.get('RESULT_ROOT')
DELAY = int(os.environ.get('DELAY'))
SPLIT = int(os.environ.get('SPLIT'))
EDGE_NAME = os.environ.get('EDGE_NAME')

isend = False
BATCH_SIZE = 5


class Reddit(Dataset):
    def __init__(self, data_root, vocab_size, vocab=None):
        
        self.users = 0
        self.num_samples = []
        self.data = []
        self.targets = []

        if vocab == None:
            counter = None
            for r in data_root:
                with open(r) as file:
                    js = json.load(file)
                    
                    counter = self.build_counter(js['user_data'], initial_counter=counter)

            if counter is not None:
                self.vocab = self.build_vocab(counter, vocab_size=vocab_size)
            else:
                print('No files to process.')
        else:
            self.vocab = vocab

        for r in data_root:
            
            with open(r) as file:
                js = json.load(file)
                self.num_samples += js['num_samples']
                for u in js['users']:
                    self.users += 1
                    if (self.users <= 280 * (SPLIT + 1)) and (self.users > 280 * SPLIT):
                        for d in js['user_data'][u]['x']:
                            for dd in d:
                                self.data.append(self.word_to_indices(dd))
                        for t in js['user_data'][u]['y']:
                            for tt in t['target_tokens']:
                                self.targets.append(self.word_to_indices(tt))
        print(len(self.data))


        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)

    def letter_to_index(self, letter):
        '''returns one-hot representation of given letter
        '''
        if letter in self.vocab.keys():
            index = self.vocab[letter]
        else:
            index = 1
        return index

    def word_to_indices(self, word):
        indices = []
        for c in word:
            if c in self.vocab.keys():
                indices.append(self.vocab[c])
            else:
                indices.append(1)

        return indices

    def build_counter(self, train_data, initial_counter=None):
        all_words = []
        for u in train_data:
            for c in train_data[u]['x']:
                for s in c:
                    all_words += s

        if initial_counter is None:
            counter = collections.Counter()
        else:
            counter = initial_counter
        counter.update(all_words)

        return counter

    def build_vocab(self, counter, vocab_size=1000):
        count_pairs = sorted(counter.items(),
                             key=lambda x: (-x[1], x[0]))  
        count_pairs = count_pairs[:(vocab_size - 2)]   

        words, counters = list(zip(*count_pairs))

        vocab = {}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1

        for i, w in enumerate(words):
            if w != '<PAD>':
                vocab[w] = i + 1

        return vocab

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        return data, target


data_root_train = ['./data/reddit/train/train_data.json']
data_root_test = ['./data/reddit/test/test_data.json']

train_data = Reddit(data_root_train, vocab_size=1000)
test_data = Reddit(data_root_test, vocab_size=1000, vocab=train_data.vocab)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

class LSTM(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=200, hidden_dim=256, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layer = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layer, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)



    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        seq_len = input.size(1)
        if hidden is None:
            h_0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim)
            c_0 = torch.zeros(self.num_layer, batch_size, self.hidden_dim)
        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(input)
        output, _ = self.lstm(embeds, (h_0, c_0))  #output size is [batch_size, seq_len, hidden_dim]
        output=self.linear(output.reshape(batch_size*seq_len,-1))
       

        return output


model = LSTM()

msgQueue = queue.Queue()

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))
def on_message(mqttc, obj, msg):
    #print("received: " + msg.topic + " " + str(msg.qos))
    if msg.topic == "Halt":
        global isend
        mqttc.unsubscribe("nondp_params/" + CLIENT_ID)  
        msgQueue.put(msg.payload)
        isend = True
    else:
        msgQueue.put(msg.payload)


client = mqtt.Client(client_id=CLIENT_ID)
#client.enable_logger(logger)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_IP, MQTT_PORT, 600)
client.subscribe([("init", 2), ("nondp_params/" + CLIENT_ID, 2), ("Halt", 2)])
client.loop_start()

if __name__ == '__main__':
    acc = []
    testloss = []
    trainloss = []
    test_idx = 1000  #test_sample's number

    params = cPickle.loads(msgQueue.get())

    for i, f in enumerate(model.parameters()):
        f.data = params[i].float()
    
    step = 0
    for epoch in range(EPOCH):

        train_loss = 0
        for idx, (data, target) in enumerate(train_loader):
         
            if isend:
                break

            model.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target.view(-1))
            train_loss += loss.item()
            loss.backward()

            if step % TEST_NUM == 0:
                print(step)
                total = 0
                correct = 0
                correct_pad = 0
                test_loss = 0
                man_file1 = open(RESULT_ROOT + '[' + str(EDGE_NAME) + ']' + '[Non-DP-Accuracy]', 'w')
                for batch_idx, (test_x, test_y) in enumerate(test_loader):
                    if batch_idx < test_idx:

                        output = model(test_x)
                        pred_y = torch.max(output, 1)[1].data.numpy()

                        test_y = test_y.view(-1)

                        test_loss += nn.CrossEntropyLoss()(output, test_y).item()
                        total += float(test_y.size(0))
                        correct += float((pred_y == test_y.data.numpy()).sum())
                        correct_pred = (pred_y == test_y.data.numpy())

                        pad = np.zeros(test_y.size(0))
                        pad_pred = (pred_y == pad)
                        correct_pad += float((correct_pred * pad_pred).sum())

                       
                    else:
                        break

                test_loss /= test_idx 
                accuracy = (correct - correct_pad) / total
                print("accuracy: ", accuracy)
                acc.append(accuracy)
                print(acc, file=man_file1)
                man_file1.close()

                print("test_loss: ", test_loss)
                man_file2 = open(RESULT_ROOT + '[' + str(EDGE_NAME) + ']' + '[Non-DP-TestLoss]', 'w')
                testloss.append(test_loss)
                print(testloss, file=man_file2)
                man_file2.close()

                
                man_file3 = open(RESULT_ROOT + '[' + str(EDGE_NAME) + ']' + '[Non-DP-TrainLoss]', 'w')
                if step != 0:
                    train_loss /= TEST_NUM
                print("train_loss: ", train_loss)
                trainloss.append(train_loss)
                print(trainloss, file=man_file3)
                man_file3.close()
                train_loss = 0
            step += 1

            grads = []
            for params in model.parameters():
                grads.append(params.grad)
            client.publish("nondp_grads/" + CLIENT_ID, cPickle.dumps(grads), 2)

            if isend:
                break
            
            params = cPickle.loads(msgQueue.get())
            for i, f in enumerate(model.parameters()):
                f.data = params[i].float()


