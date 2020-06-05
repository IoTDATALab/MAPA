#coding=utf-8
import torch
import syft as sy  # <-- NEW: import the Pysyft library
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from visdom import Visdom
from datetime import datetime
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
import ComputePrivacy as Privacy#导入自定义计算隐私预算函数
import Datasets
import os
import logging

logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')
vis = Visdom(env='FEMNIST_NonDP_ASyn')

# Define parameters
class Arguments():
    def __init__(self):
        self.batch_size = 8  # Number of samples used of each user/device at each iteration.
        # If this value is less than 1, then it means the sampling ratio, else it means the mini-batch size
        self.lr = 0.2  # Learning rate
        self.users_total = 180  # Total number of users/devices
        self.user_sel_prob = 0.05  # Probability for sampling users/devices at each iteration
        self.itr_numbers = 10000  # Number of total iterations

        self.test_batch_size = 128  # Number of test mini-batch size
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

# Define model
class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

###################################################################################
############################### Define functions ##################################

################### Define split of users/devices ##############
def Virtual_Users_num(Leaf_split, LEAF=True):
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
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    model.train()
    model.zero_grad()
    output = model(train_data)
    loss = F.nll_loss(output, train_target)
    loss.backward()
    Gradients_Tensor = []
    if gradient == False:
        for params in model.parameters():
            Gradients_Tensor.append(-learning_rate*params.grad.data)
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss +=  F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_loader_len
    test_acc = correct / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc

#################################################################
######################Load Dataset##############
data_root_train = []
data_root_test = []
for i in range(0,4):
    data_root_train.append('../data/FEMINIST/train/all_data_' + str(i) + '_niid_0_keep_0_train_9.json')
    data_root_test.append('../data/FEMINIST/test/all_data_' + str(i) + '_niid_0_keep_0_test_9.json')
#Load train datasets
train_loader = Datasets.Femnist(data_root_train, user_size=args.users_total)
Leaf_split = train_loader.num_samples
output_size = len(set(train_loader.targets))
print('Users number is {}, trainning set size is {}, the least samples per user is {}, output size is {} \n split is {}'.
      format(len(Leaf_split), len(train_loader), min(Leaf_split),output_size, Leaf_split ))
batch_size = min(min(Leaf_split), args.batch_size)
#Load test datasets
test_loader = torch.utils.data.DataLoader(Datasets.Femnist(data_root_test, user_size=args.users_total),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

######################################################################
##############Generate model and users##############
print('The number of classes is {}'.format(output_size))
model = Net(output_size=output_size).to(device)
workers = []
models = {}
optims = {}
Users_num_total, Ratio = Virtual_Users_num(Leaf_split, LEAF=True)
for i in range(1, Users_num_total+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))
    # exec('workers["user{}"] = user{}'.format(i,i))
optim_sever = optim.SGD(params=model.parameters(),lr=args.lr)

########## Assign train dataset to all users/devices ############
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio, net='CNN')
# Criteria = nn.CrossEntropyLoss()

######## Initial model accuracy on test dataset #######
test_loss, test_acc = test(model, device, test_loader)

######################################################################
##############Federated learning process##############
#Define visdom
logs = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'varepsilon': []}
logs['test_acc'].append(test_acc)
logs['test_loss'].append(test_loss)
Results_testloss = vis.line(np.array([test_loss]), [1], win='Test_loss',
                            opts=dict(title='Test loss on Sent140', legend=['Test loss']))
Results_testacc = vis.line(np.array(np.array([test_acc])), [1], win='Test_acc',
                            opts=dict(title='Test accuracy on Sent140', legend=['Test accuracy']))
Results_trainloss = vis.line([0.], [1], win='Train_acc',
                            opts=dict(title='Train loss on Sent140', legend=['Train loss']))

###################################################################################
#################################联邦学习过程######################################
# Obtain information of layers
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
# Set learning rate
lr = args.lr

# Define train/test process
for itr in range(1, args.itr_numbers + 1):
    #Select the participants from the total users with the given probability
    Users_Current = np.random.binomial(Users_num_total, args.user_sel_prob, 1).sum()
    if Users_Current == 0:
        Users_Current = 1

    # Load samples from the participants with the given probability or mini-batch size args.batch_size
    federated_train_loader = sy.FederatedDataLoader(Federate_Dataset, batch_size=args.batch_size, shuffle=True,
                                                    worker_num=Users_Current, batch_num=args.batchs_round, **kwargs)
    workers_list = federated_train_loader.workers  # List of participants at the current iteration

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
        train_data, train_targets = train_data.get(), train_targets.get()

        Gradients_Tensor, loss = train(lr, model_round, train_data, train_targets, device, optimizer, gradient=True)
        Layers_num = len(Gradients_Tensor)
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
    for idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-lr, Collect_Gradients[idx])

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
        if itr == 1:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='replace')
        else:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='append')

    ############ Print test loss #################
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)
        vis.line(np.array(np.array([test_loss])), np.array([itr]), win=Results_testloss, update='append')
        vis.line(np.array(np.array([test_acc])), np.array([itr]), win=Results_testacc, update='append')

with open('../results/FEMNIST/NonDP_Asyn_TestLoss.txt', 'a+') as fl:
    fl.write('\n {} Results  (UN is {}, SP is {}, BZ is {}, LR is {})\n'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr))
    fl.write(str(test_loss) + '\t')
with open('../results/FEMNIST/NonDP_Asyn_TestAcc.txt', 'a+') as fl:
    fl.write('\n %{} Results  (UN is {}, SP is {}, BZ is {}, LR is {})\n'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr))
    fl.write(str(test_acc) + '\t')
with open('../results/FEMNIST/NonDP_Asyn_TrainLoss.txt', 'a+') as fl:
    fl.write('\n {} Results (UN is {}, SP is {}, BZ is {}, LR is {})\n'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr))

with open('../results/FEMNIST/NonDP_Asyn_TrainLoss.txt', 'a+') as fl:
    fl.write('train_loss: '+str(logs['train_loss']))
with open('../results/FEMNIST/NonDP_Asyn_TestLoss.txt', 'a+') as fl:
    fl.write('test_loss: '+str(logs['test_loss']))
with open('../results/FEMNIST/NonDP_Asyn_TestAcc.txt', 'a+') as fl:
    fl.write('test_acc: '+str(logs['test_acc']))

def modelSave():
    dir = "./model/NonDP/"+'lr'+str(args.lr)+'bz'+str(args.batch_size)+'itr'+str(args.itr_numbers)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save({
        'args': args,
        'Leaf_split': Leaf_split,
        'data': date
    }, dir+'/params')
    torch.save({
        'model': model,
        'optim_server': optim_sever,
        'models': models,
        'optims': optims
    }, dir+'/models')
if args.save_model:
    modelSave()