import torch
import syft as sy  # <-- NEW: import the Pysyft library
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from visdom import Visdom
from torchvision import datasets, transforms
from datetime import datetime
import ComputePrivacy as Privacy # Import self definition function to compute the privacy loss
import logging
import Datasets # Import self definition function to load the federated datasets
import os
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')
# vis = Visdom(env='SYNTHETIC_FixDP_S_Asyn_04flat')

# Define parameters
class Arguments():
    def __init__(self):
        self.batch_size = 0.05 # Number of samples used of each user/device at each iteration.
        # If this value is less than 1, then it means the sampling ratio, else it means the mini-batch size
        self.lr = 0.1  # Learning rate
        self.clipbound = torch.tensor([3.])  # clipbound
        self.z = 0.4  # Noise parameter z in Gaussian noise N(0, (zS)^2) where S is sensitivity
        self.users_total = 1000  # Total number of users/devices
        self.user_sel_prob = 0.1  # Probability for sampling users/devices at each iteration
        self.itr_numbers = 10000  # Number of total iterations

        self.test_batch_size = 100  # Number of test mini-batch size
        self.log_train = 50  # Logging interval for printing the training loss
        self.log_test = 50  # Logging interval for printing the test accuracy
        self.save_model = False
        self.batchs_round = 1  # Number of mini-batchs of each selected user in each iteration
        self.no_cuda = True
        self.seed = 1
        self.ClipStyle = 'Flat'  # Clipping method, including Flat and Per-Layer
        self.Leaf = True # Use the split provided by LEAF

args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lr = nn.Linear(60, 5)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x

###################################################################################
############################### Define functions ##################################

################### Define split of users/devices ##############
def Virtual_Users_num(Leaf_split, LEAF=True, Skew=0):
    Users_num_total = len(Leaf_split)
    if LEAF:
        Ratio = Leaf_split
    else:
        Users_num_total = 50
        if Skew == 0:
            Ratio = [random.randint(1, 10) for _ in range(Users_num_total)]
        if Skew == 1:
            Ratio = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10
        if Skew == 2:
            Ratio = [1]*10 + [5]*10 + [10]*10 + [15]*10 + [20]*10
        if Skew == 3:
            Ratio = [1]*10 + [50]*10 + [100]*10 + [150]*10 + [200]*10

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

#######################Define Clipping#######################
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

################ Dfine adding noise###################
def AddNoise(Tensor, Clipboud):
    std = args.z
    if len(Clipboud) == 1:
        for i in range(len(Tensor)):
            Tensor[i] = Tensor[i] + std * Clipboud[0] * torch.randn(Tensor[i].shape)
    if len(Clipboud) > 1:
        for i in range(len(Tensor)):
            Tensor[i] = Tensor[i] + std * Clipboud[i] * torch.randn(Tensor[i].shape)
    return Tensor

##########################Define training process########################
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    model.train()
    model.zero_grad()
    output = model(train_data.float())
    loss = Criteria(output, train_target)
    loss.backward()
    Gradients_Tensor = []
    if gradient == False:
        for params in model.parameters():
            Gradients_Tensor.append(-learning_rate*params.grad.data)
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
            data, target = data.to(device), target.to(device)
            output = model(data.float()).squeeze()
            test_loss += Criteria(output, target).item() # sum up batch loss
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc

#################################################################
######################Load Dataset##############
# Path
data_root_train = '../data/synthetic/train/data_niid_0_keep_5_train_6.json'
data_root_test = '../data/synthetic/test/data_niid_0_keep_5_test_6.json'
# Load train dataset
train_loader = Datasets.synthetic(data_root_train, user_size=args.users_total)
Leaf_split = train_loader.num_samples
output_size = len(set(train_loader.targets))
print('Users number is {}, trainning set size is {}, the least samples per user is {}, output size is {} \n split is {}'.
      format(len(Leaf_split), len(train_loader), min(Leaf_split),output_size, Leaf_split ))

# Load test dataset
test_loader = torch.utils.data.DataLoader(Datasets.synthetic(data_root_test, user_size=args.users_total),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

###################################################################################
##############Generate model and users##############
model = Net().to(device)
workers = []
models = {}
optims = {}
Users_num_total, Ratio = Virtual_Users_num(Leaf_split, LEAF=args.Leaf, Skew=0)
print(Ratio, Users_num_total)

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
########Initial model accuracy on test dataset#######
test_loss, test_acc = test(model, device, test_loader)

# Logging files for experiment results
with open('../results/SYNTHETIC/FixDP_Asyn_TestLoss_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, batch_size, args.lr, args.z, args.clipbound,
                    args.ClipStyle, args.Leaf))
    fl.write(str(test_loss) + '\t')
with open('../results/SYNTHETIC/FixDP_Asyn_TestAcc_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, batch_size, args.lr, args.z, args.clipbound,
                    args.ClipStyle, args.Leaf))
    fl.write(str(test_acc) + '\t')
with open('../results/SYNTHETIC/FixDP_Asyn_TrainLoss_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, batch_size, args.lr, args.z, args.clipbound,
                    args.ClipStyle, args.Leaf))
with open('../results/SYNTHETIC/FixDP_Asyn_Budget_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, batch_size, args.lr, args.z, args.clipbound,
                    args.ClipStyle, args.Leaf))

# Logging dictionary
logs = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'varepsilon': []}

# Visdom results
# Results_testloss = vis.line([test_loss], [1], win='Test_loss',
#                             opts=dict(title='Test loss on Synthetic', legend=['Test loss']))
# Results_testacc = vis.line([test_acc], [1], win='Test_acc',
#                             opts=dict(title='Test accuracy on Synthetic', legend=['Test accuracy']))
# Results_trainloss = vis.line([0.], [1], win='Train_acc',
#                             opts=dict(title='Train loss on Synthetic', legend=['Train loss']))
# Results_varepsilon = vis.line([0.], [1], win='varepsilon',
#                             opts=dict(title='Pirvacy budget on Synthetic', legend=['Test accuracy']))

###################################################################################
##############Federated learning process##############

# Obtain information of layers
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
# Set learning rate
lr = args.lr
# Set clipping bound
Clip_Bound = ClipBound_gerate(args.clipbound, Layers_nodes, style=args.ClipStyle)
# Compute the sampling ratio for moment account
Sampled_ratio = args.user_sel_prob * args.batch_size
delta = 1. / sum(Leaf_split) ** 1.1 # delta in (varepsilon, delta)-DP

# Privacy loss of the first iteraiton
varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, 1, delta, 32)
logs['varepsilon'].append(varepsilon)
with open('../results/SYNTHETIC/FixDP_Asyn_Budget_04flat.txt', 'a+') as fl:
    fl.write(str(varepsilon) + '\t')
# vis.line(np.array([varepsilon]), np.array([1]), win=Results_varepsilon, update='replace')

# Define train/test process
for itr in range(1, args.itr_numbers + 1):
    #Select the participants from the total users with the given probability
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
        train_data, train_targets = train_data.get(), train_targets.get()

        Collect_Gradients_Per_user = ZerosGradients(Layers_shape)
        Loss_Per_user = torch.tensor(0.)
        ## Compute and Clip gradients per sample
        for inner_dex in range(train_data.size(0)):
            inner_data = torch.unsqueeze(train_data[inner_dex], 0)
            inner_target = torch.unsqueeze(train_targets[inner_dex], 0)
            Gradients_Sample, Loss_Sample = train(lr, model_round, inner_data, inner_target, device, optimizer,
                                                  gradient=True)
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

    # Send the updated model to the corresponding participants
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
        with open('../results/SYNTHETIC/FixDP_Asyn_TrainLoss_04flat.txt', 'a+') as fl:
            fl.write(str(Loss_train.item()) + '\t')
        # if itr == 1:
        #     vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='replace')
        # else:
        #     vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='append')

    ############ Print test loss #################
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)
        varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, itr, delta, 32)
        print('Privacy Budget: {:.6f}; {} -th Iteration'.format(varepsilon, itr))
        logs['varepsilon'].append(varepsilon)
        with open('../results/SYNTHETIC/FixDP_Asyn_TestLoss_04flat.txt', 'a+') as fl:
            fl.write(str(test_loss) + '\t')
        with open('../results/SYNTHETIC/FixDP_Asyn_TestAcc_04flat.txt', 'a+') as fl:
            fl.write(str(test_acc) + '\t')
        with open('../results/SYNTHETIC/FixDP_Asyn_Budget_04flat.txt', 'a+') as fl:
            fl.write(str(varepsilon) + '\t')
        # vis.line(np.array([test_loss]), np.array([itr]), win=Results_testloss, update='append')
        # vis.line(np.array([test_acc]), np.array([itr]), win=Results_testacc, update='append')
        # vis.line(np.array([varepsilon]), np.array([itr]), win=Results_varepsilon, update='append')
