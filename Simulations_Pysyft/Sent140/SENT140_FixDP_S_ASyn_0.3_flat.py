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

logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')
vis = Visdom(env='SENT140_FixedDP_ASyn')
#定义参量
class Arguments():
    def __init__(self):

        self.batch_size = 0.3 # Number of samples used of each user/device at each iteration.
        # If this value is less than 1, then it means the sampling ratio, else it means the mini-batch size
        self.lr = 0.1  # Learning rate
        self.grad_upper_bound = torch.tensor([0.4])  # clipbound
        self.z = 0.3  # Noise parameter z in Gaussian noise N(0, (zS)^2) where S is sensitivity
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
    def __init__(self, input_size=50, hidden_size=120, output_size=2, num_layer=3):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, hidden = self.lstm(input)
        output, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.linear(output[:, -1, :])
        # output = self.linear(output.contiguous().view(output.shape[0], -1))
        return output

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layer, 10, self.hidden_size))

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

################## Define Clipping bound for Flat/Per-Layer ###############
def ClipBound_gerate(Clipbound, Layers_nodes, style="Flat"):
    Layer_nodes = torch.tensor(Layers_nodes).float()
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

####################### Define Clipping #######################
def TensorClip(Tensor, ClipBound):
    norm = torch.tensor([0.])
    Layers_num = len(Tensor)
    Tuple_num = torch.ones(Layers_num)
    for i in range(Layers_num):
        norm = norm + Tensor[i].float().norm()**2
        Tuple_num[i] = Tensor[i].numel()
    norm = norm.sqrt()
    if len(ClipBound) == 1:
        Indicator = 1*(norm<=ClipBound[0])
        for i in range(Layers_num):
            Tensor[i] = Tensor[i]*torch.min(torch.ones(1), ClipBound[0]/norm)
    if len(ClipBound) > 1:
        Indicator = torch.zeros(Layers_num)
        for i in range(Layers_num):
            norm_layer = Tensor[i].float().norm()
            Indicator[i] = 1*(norm_layer<=ClipBound[i])
            Tensor[i] = Tensor[i]*torch.min(torch.ones(1), ClipBound[i]/norm_layer)
    return Tensor, Indicator

################ Dfine adding noise ###################
def AddNoise(Tensor, Clipboud):
    std = args.z
    if len(Clipboud) == 1:
        for i in range(len(Tensor)):
            Tensor[i] = Tensor[i] + std * Clipboud[0] * torch.randn(Tensor[i].shape)
    if len(Clipboud) > 1:
        for i in range(len(Tensor)):
            Tensor[i] = Tensor[i] + std * Clipboud[i] * torch.randn(Tensor[i].shape)
    return Tensor

########################## Define training process ########################
def train(learning_rate, model, train_data, train_target, gradient=True):
    model.train()
    model.zero_grad()
    output = model(train_data)
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


########################### Define test function ########################
def test(model, device, test_loader):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data, dim=1)
            batch_size = data.shape[0]

            _, idx_sort = torch.sort(target[1], dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            data = data.index_select(0, idx_sort)
            target[1] = target[1].index_select(0, idx_sort)

            data = torch.nn.utils.rnn.pack_padded_sequence(data, target[1], batch_first=True)
            output = model(data)
            output = output.index_select(0, idx_unsort)
            test_loss +=  Criteria(output, target[0]) * batch_size # sum up batch loss
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target[0].data.view_as(pred)).long().cpu().sum()

    test_loss /= test_loader_len
    test_acc = correct.float() / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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


###################################################################################
##########Assign train dataset to all users/devices############
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio)
Criteria = nn.CrossEntropyLoss()

###################################################################################
############# Logging files ############
test_loss, test_acc = test(model, device, test_loader) # The prediction accuracy of the initial model
with open('../results/SENT140/FixedDP_Asyn_TestLoss.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Flat, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))
    fl.write(str(test_loss) + '\t')
with open('../results/SENT140/FixedDP_Asyn_TestAcc.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Flat, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))
    fl.write(str(test_acc) + '\t')
with open('../results/SENT140/FixedDP_Asyn_TrainLoss.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Flat, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))
with open('../results/SENT140/FixedDP_Asyn_Budget.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Flat, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))

###################################################################################
#################################Define visualization results####################################
#Define recording dictionary
logs = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'varepsilon': []}
Results_testloss = vis.line(np.array([test_loss.numpy()]), [1], win='Test_loss',
                            opts=dict(title='Test loss on Sent140', legend=['Test loss']))
Results_testacc = vis.line(np.array(np.array([test_acc.numpy()])), [1], win='Test_acc',
                            opts=dict(title='Test accuracy on Sent140', legend=['Test accuracy']))
Results_trainloss = vis.line([0.], [1], win='Train_acc',
                            opts=dict(title='Train loss on Sent140', legend=['Train loss']))
Results_varepsilon = vis.line([0.], [1], win='varepsilon',
                            opts=dict(title='Pirvacy budget on Sent140', legend=['Test accuracy']))

###################################################################################
##############Federated learning process##############
# Obtain information of layers
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
# Set learning rate
lr = args.lr
# Set clipping bound
Clip_Bound = ClipBound_gerate(args.grad_upper_bound, Layers_nodes, style=args.ClipStyle)
# Set clipping bound
# Compute the sampling ratio for moment account
Sampled_ratio = args.user_sel_prob * args.batch_size
delta = 1. / sum(Leaf_split) ** 1.1 # delta in (varepsilon, delta)-DP

# Privac loss of the first iteration
varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, 1, delta, 32)
logs['varepsilon'].append(varepsilon)
with open('../results/SENT140/FixedDP_Asyn_Budget.txt', 'a+') as fl:
    fl.write(str(varepsilon) + '\t')
vis.line(np.array([varepsilon]), np.array([1]), win=Results_varepsilon, update='replace')

# Define train/test process
for itr in range(1, args.itr_numbers + 1):
    #Select the participants from the total users with the given probability
    Users_Current = np.random.binomial(Users_num_total, args.user_sel_prob, 1).sum()
    if Users_Current == 0:
        Users_Current = 1

    # Load samples from the participants with the given probability or mini-batch size args.batch_size
    federated_train_loader = sy.FederatedDataLoader(Federate_Dataset, batch_size=batch_size, shuffle=True,
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
        train_data, train_targets = train_data.get(), train_targets.get()

        train_targets, length = torch.split(train_targets, 1, dim=1)#split target and length

        Collect_Gradients_Per_user = ZerosGradients(Layers_shape)
        Loss_Per_user = torch.tensor(0.)
        ## Compute and Clip gradients per sample
        for inner_dex in range(train_data.size(0)):
            inner_data = torch.unsqueeze(train_data[inner_dex], 0)
            inner_data = torch.nn.utils.rnn.pack_padded_sequence(inner_data, length[inner_dex], batch_first=True)
            inner_target = train_targets[inner_dex]
            Gradients_Sample, Loss_Sample = train(lr, model_round, inner_data, inner_target, gradient=True)
            Gradients_Sample, _ = TensorClip(Gradients_Sample, Clip_Bound)
            Loss_Per_user += Loss_Sample
            for i in range(Layers_num):
                Collect_Gradients_Per_user[i] += Gradients_Sample[i]
        # Average gradients
        for i in range(Layers_num):
            Collect_Gradients_Per_user[i] /= (inner_dex + 1)
        Loss_Per_user /= (inner_dex + 1)
        Collect_Gradients_Per_user = AddNoise(Collect_Gradients_Per_user, Clip_Bound / (inner_dex + 1))
        Loss_train += Loss_Per_user

        for i in range(Layers_num):
            Collect_Gradients[i] += Collect_Gradients_Per_user[i]
    for grad_idx, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(-lr, Collect_Gradients[grad_idx] / (idx_outer + 1))
    Loss_train /= (idx_outer + 1)

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
        with open('../results/SENT140/FixedDP_Asyn_TrainLoss.txt', 'a+') as fl:
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
        varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, itr, delta, 32)
        print('Privacy Budget: {:.6f}; {} -th Iteration'.format(varepsilon, itr))
        logs['varepsilon'].append(varepsilon)
        with open('../results/SENT140/FixedDP_Asyn_TestLoss.txt', 'a+') as fl:
            fl.write(str(test_loss) + '\t')
        with open('../results/SENT140/FixedDP_Asyn_TestAcc.txt', 'a+') as fl:
            fl.write(str(test_acc) + '\t')
        with open('../results/SENT140/FixedDP_Asyn_Budget.txt', 'a+') as fl:
            fl.write(str(varepsilon) + '\t')
        vis.line(np.array(np.array([test_loss.numpy()])), np.array([itr]), win=Results_testloss, update='append')
        vis.line(np.array(np.array([test_acc.numpy()])), np.array([itr]), win=Results_testacc, update='append')
        vis.line(np.array([varepsilon]), np.array([itr]), win=Results_varepsilon, update='append')