import torch
import syft as sy  # <-- NEW: import the Pysyft library
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
from torch.autograd import Variable
from visdom import Visdom
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
import ComputePrivacy as Privacy# Import self definition function to compute the privacy loss
import Datasets
import os
import logging
import time

logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')
vis = Visdom(env='SENT140_NonDP_ASyn')

#定义参量
class Arguments():
    def __init__(self):
        self.batch_size = 0.3 # Number of samples used of each user/device at each iteration.
        # If this value is less than 1, then it means the sampling ratio, else it means the mini-batch size
        self.lr = 0.01  # Learning rate, is adaptively adjusted
        self.z = 0.1  # Noise parameter z in Gaussian noise N(0, (zS)^2) where S is sensitivity
        self.users_total = 1000  # Total number of users/devices
        self.user_sel_prob = 0.01  # Probability for sampling users/devices at each iteration
        self.itr_numbers = 5000  # Number of total iterations

        self.test_batch_size = 1  # Number of test mini-batch size
        self.log_train = 100  # Logging interval for printing the training loss
        self.log_test = 100  # Logging interval for printing the test accuracy
        self.save_model = False
        self.batchs_round = 1  # Number of mini-batchs of each selected user in each iteration
        self.no_cuda = True
        self.seed = 1
        self.ClipStyle = 'Flat'  # Clipping method, including Flat and Per-Layer



args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#定义模型
class Net(nn.Module):
    def __init__(self, input_size=50, hidden_size=120, output_size=2, num_layer=1):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, batch_size):
        h_0 = Variable(torch.zeros(self.num_layer, batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layer, batch_size, self.hidden_size))
        output, (h_0, c_0) = self.lstm(input, (h_0, c_0))
        # output, self.hidden = self.lstm(input, self.hidden)
        output, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.linear(output[:, -1, :])
        # output = self.linear(output.contiguous().view(output.shape[0], -1))
        return output

    # def init_hidden(self):
    #     return Variable(torch.zeros(self.num_layer, 10, self.hidden_size))

###################################################################################
############################### Define functions ############################################

################################ Define split of users/devices #######################
def Virtual_Users_num(Leaf_split, LEAF=True):
    Users_num_total = 0
    if LEAF:
        Users_num_total = len(Leaf_split)
        Ratio = Leaf_split
    else:
        Users_num_total = args.users_total
        Ratio = [random.randint(1, 10) for _ in range(Users_num_total)]
    return Users_num_total, Ratio

############ Obtain the number and shape of layers of the model ##########
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes

######################## Initialize all layers as zero ###################
def ZerosGradients(Layers_shape):
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i]))
    return ZeroGradient

########################## Define training process ########################
def ClipBound_gerate(Clipbound, Layers_nodes, style="Flat"):
    Layer_nodes = torch.tensor(Layers_nodes).float()
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

########################## Define training process ########################
def train(learning_rate, model, train_data, train_target, idx_unsort, batch_size, gradient=True):
    model.train()
    model.zero_grad()
    output = model(train_data, batch_size)
    output = output.index_select(0,idx_unsort)
    loss = Criteria(output, train_target)
    loss.backward()
    Gradients_Tensor = []
    if gradient == False:
        for params in model.parameters():
            Gradients_Tensor.append(-learning_rate*params.grad.data)#return -lr*grad
    if gradient == True:
        for params in model.parameters():
            Gradients_Tensor.append(params.grad.data)
    return Gradients_Tensor, loss

########################### Define test function #############################
def test(model, device, test_loader):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data, dim=1)
            # batch_size = data.shape[0]

            _, idx_sort = torch.sort(target[1], dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            data = data.index_select(0, idx_sort)
            target[1] = target[1].index_select(0, idx_sort)

            data = torch.nn.utils.rnn.pack_padded_sequence(data, target[1], batch_first=True)
            output = model(data, args.test_batch_size)
            output = output.index_select(0, idx_unsort)
            test_loss +=  Criteria(output, target[0]) * args.test_batch_size # sum up batch loss
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target[0].data.view_as(pred)).long().cpu().sum()

    test_loss /= test_loader_len
    test_acc = correct.float() / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc

###################################################################################
######################Load Dataset##############
#Path
vocab_root = '../data/sentiment140/embs.json'
data_root_train = "../data/sentiment140/train/all_data_niid_0_keep_0_train_9.json"
data_root_test = "../data/sentiment140/test/all_data_niid_0_keep_0_test_9.json"
#Load train datasets
train_loader = Datasets.sentiment140(data_root_train, vocab_root, user_size=args.users_total)
Leaf_split = train_loader.num_samples# original split provided by LEAF
output_size = len(set(train_loader.targets))
print('Users number is {}, trainning set size is {}, the least samples per user is {}, output size is {} \n split is {}'.
      format(len(Leaf_split), len(train_loader), min(Leaf_split),output_size, Leaf_split ))
batch_size = min(min(Leaf_split), args.batch_size)

# Load train datasets
test_loader = torch.utils.data.DataLoader(Datasets.sentiment140(data_root_test,vocab_root, user_size=args.users_total),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

###################################################################################
##############Generate model and users##############
model = Net().to(device)
workers = []
models = {}
optims = {}
Users_num_total, Ratio = Virtual_Users_num(Leaf_split, LEAF=True)
for i in range(1, Users_num_total+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))
optim_sever = optim.SGD(params=model.parameters(),lr=args.lr)
print('Total users number is {}, the minimal number of per user is {}'.format(Users_num_total, min(Leaf_split)))

###################################################################################
##########Assign train dataset to all users/devices############
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio)
Criteria = nn.CrossEntropyLoss()

###################################################################################
############# Logging files ############
test_loss, test_acc = test(model, device, test_loader) # The prediction accuracy of the initial model
with open('../results/SENT140/NonDP_Asyn_TestLoss.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Flat, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z))
    fl.write(str(test_loss) + '\t')
with open('../results/SENT140/NonDP_Asyn_TestAcc.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Flat, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z))
    fl.write(str(test_acc) + '\t')
with open('../results/SENT140/NonDP_Asyn_TrainLoss.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Flat, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z))

###################################################################################
#################################Define visualization results####################################
#Define recording dictionary
logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
Results_testloss = vis.line(np.array([test_loss.numpy()]), [1], win='Test_loss',
                            opts=dict(title='Test loss on Sent140', legend=['Test loss']))
Results_testacc = vis.line(np.array(np.array([test_acc.numpy()])), [1], win='Test_acc',
                            opts=dict(title='Test accuracy on Sent140', legend=['Test accuracy']))
Results_trainloss = vis.line([0.], [1], win='Train_acc',
                            opts=dict(title='Train loss on Sent140', legend=['Train loss']))

###################################################################################
############## Federated learning process ##############

# Obtain information of layers
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
# Set learning rate
lr = args.lr
# Define train/test process
for itr in range(1, args.itr_numbers + 1):
    # Select the participants from the total users with the given probability
    Users_Current = np.random.binomial(Users_num_total, args.user_sel_prob, 1).sum()
    if Users_Current == 0:
        Users_Current = 1
    # Load samples from the participants with the given probability or mini-batch size args.batch_size
    federated_train_loader = sy.FederatedDataLoader(Federate_Dataset, batch_size=args.batch_size, shuffle=True,
                                                    worker_num=Users_Current, batch_num=args.batchs_round, **kwargs)
    workers_list = federated_train_loader.workers # List of participants at the current iteration

    # Next two lines are only necessary for synchronous aggregation
    # for idx in range(len(workers_list)):
    #     models[workers_list[idx]] = model

    # Initialize the same model-structure tensor with zero elements
    Collect_Gradients = ZerosGradients(Layers_shape)
    Loss_train = torch.tensor(0.)
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = Variable(train_data.get()), Variable(train_targets.get())

        train_targets, length = torch.split(train_targets, 1, dim=1)  # split target and length
        length = torch.squeeze(length, dim=1)
        train_targets = torch.squeeze(train_targets, dim=1)

        # Sort data according to sentence length
        _, idx_sort = torch.sort(length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        train_data = train_data.index_select(0, idx_sort)
        length = length.index_select(0, idx_sort)

        batch_size = train_data.size(0)

        train_data = torch.nn.utils.rnn.pack_padded_sequence(train_data, length, batch_first=True)  # pack
        Gradients_Tensor, loss = train(lr, model_round, train_data, train_targets, idx_unsort, batch_size, gradient=True)
        Loss_train += loss
        # Accumulate gradients for participants
        for i in range(Layers_num):
            Collect_Gradients[i] = Collect_Gradients[i] + Gradients_Tensor[i]
    # Average gradients
    for i in range(Layers_num):
        Collect_Gradients[i] = Collect_Gradients[i] / (idx_outer + 1)
    # Average loss
    Loss_train /= (idx_outer + 1)
    # Update the global model using conventional gradient descent
    for idx_para, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-lr, Collect_Gradients[idx_para])

    # Send the updated model to the corresponding participants, only for asynchronous aggregation
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for _, (params_server, params_client) in enumerate(zip(model.parameters(),worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model

    ############ Print train loss #################
    if itr == 1 or itr % args.log_train == 0:
        print('Train Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            itr, itr, args.itr_numbers,
            100. * itr / args.itr_numbers, Loss_train.item()))
        logs['train_loss'].append(Loss_train.item())
        with open('../results/SENT140/NonDP_Asyn_TrainLoss.txt', 'a+') as fl:
            fl.write(str(Loss_train.item()) + '\t')
        if itr == 1:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='replace')
        else:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='append')

        ############ Print test loss #################
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)
        with open('../results/SENT140/NonDP_Asyn_TestLoss.txt', 'a+') as fl:
            fl.write(str(test_loss) + '\t')
        with open('../results/SENT140/NonDP_Asyn_TestAcc.txt', 'a+') as fl:
            fl.write(str(test_acc) + '\t')
        vis.line(np.array(np.array([test_loss.numpy()])), np.array([itr]), win=Results_testloss, update='append')
        vis.line(np.array(np.array([test_acc.numpy()])), np.array([itr]), win=Results_testacc, update='append')