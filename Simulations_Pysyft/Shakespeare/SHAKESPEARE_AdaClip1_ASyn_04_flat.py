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
# from visdom import Visdom
from datetime import datetime
import ComputePrivacy as Privacy # Import self definition function to compute the privacy loss
import Datasets # Import self definition function to load the federated datasets
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
date = datetime.now().strftime('%Y-%m-%d %H:%M')
# vis = Visdom(env='Shakes_AdaClip1_ASyn')

# Define parameters
class Arguments():
    def __init__(self):
        self.batch_size = 0.002 # Number of samples used of each user/device at each iteration.
        # If this value is less than 1, then it means the sampling ratio, else it means the mini-batch size
        self.lr = 0.1 # Learning rate
        self.ClipBound = torch.tensor([0.01]) # clipbound
        self.z = 0.4    # Noise parameter z in Gaussian noise N(0, (zS)^2) where S is sensitivity
        self.users_total = 118    # Total number of users/devices
        self.user_sel_prob = 0.1    # Probability for sampling users/devices at each iteration
        self.itr_numbers = 3000    # Number of total iterations

        self.test_batch_size = 10000 # Number of test mini-batch size
        self.log_train = 100 # Logging interval for printing the training loss
        self.log_test = 100 # Logging interval for printing the test accuracy
        self.save_model = False
        self.batchs_round = 1    # Number of mini-batchs of each selected user in each iteration
        self.no_cuda = True
        self.seed = 1
        self.ClipStyle = 'Flat' # Clipping method, including Flat and Per-Layer
        self.Leaf = True # Use the split provided by LEAF
        self.beta_1 = 0.99     # Averaging speed parameter for the mean of gradients
        self.beta_2 = 0.9      # Averaging speed parameter for the variance of gradients
        self.h_1 = torch.tensor(10**(-10)) # Parameter 1 for estimating stand variance
        self.h_2 = torch.tensor(10**(-5)) # Parameter 2 for estimating stand variance


args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Define model
class Net(nn.Module):
    def __init__(self, input_size=8, hidden_size=256, output_size=79, num_layer=2):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.embedding = nn.Embedding(79, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        decoder = self.embedding(input)
        output, hidden = self.lstm(decoder)
        output = self.linear(output[:, -1, :])
        # output = self.linear(output.contiguous().view(output.shape[0], -1))
        return output

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layer, 10, self.hidden_size))

###################################################################################
############################## Define functions ###################################

################### Define split of users/devices ##############
def Virtual_Users_num(Leaf_split, LEAF=True, Skew=0):
    Users_num_total = len(Leaf_split)
    if LEAF == True:
        Ratio = Leaf_split
    else:
        Users_num_total = 50
        #生成用户数据切分比例
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

####### Initialize expectation M_0 and standard variance S_0 #############
def InitializeMS(Layers_shape):
    M_0 = []
    S_0 = []
    for i in range(len(Layers_shape)):
        M_0.append(torch.zeros(Layers_shape[i]))
        S_0.append(torch.sqrt(torch.ones(1)*args.h_1*args.h_2)*torch.ones(Layers_shape[i]))
    return M_0, S_0

########################### Compute B based on S ##########################
def ComputeB(S):
    S_sum = torch.zeros(1)
    S_len = len(S)
    B = []
    for i in range(S_len):
        S_sum = S_sum + S[i].sum()
    for i in range(S_len):
        B.append(torch.sqrt(S[i])*torch.sqrt(S_sum))
    return B

############################### Update M and S ############################
def UpdateMS(Gradients, M, S, B, args, Layers_num, Clip_Bound):
    if len(Clip_Bound) ==1:
        for i in range(Layers_num):
            M[i] = args.beta_1 * M[i] + (1 - args.beta_1) * Gradients[i]
            temp = torch.min(torch.max(((Gradients[i] - M[i])**2 - B[i]**2 * (args.z*Clip_Bound[0])**2), args.h_1.float()),
                             args.h_2.float())
            S[i] = torch.sqrt(args.beta_2 * S[i]**2 + (1 - args.beta_2)*temp)
    if len(Clip_Bound) >1:
        for i in range(Layers_num):
            M[i] = args.beta_1 * M[i] + (1 - args.beta_1) * Gradients[i]
            temp = torch.min(torch.max(((Gradients[i] - M[i])**2 - B[i]**2 * (args.z*Clip_Bound[i])**2), args.h_1.float()),
                             args.h_2.float())
            S[i] = torch.sqrt(args.beta_2 * S[i]**2 + (1 - args.beta_2)*temp)
    return M, S

################## Define Clipping bound for Flat/Per-Layer ###############
def ClipBound_gerate(Clipbound, Layers_nodes, style="Flat"):
    Layer_nodes = torch.tensor(Layers_nodes).float()
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

############# Define clipping and adding noise for AdaClip1 #################
def Noise_Addition(Layers_num, Layers_shape, Gradients, M, B, ClipBound):
    Gradients_norm = torch.tensor([0.])
    variance = args.z
    for i in range(Layers_num):
        Gradients[i] = (Gradients[i] - M[i])/B[i]
        Gradients_norm = Gradients_norm + Gradients[i].norm()**2
    Gradients_norm = Gradients_norm.sqrt()

    if len(ClipBound) > 1:
        for i in range(Layers_num):
            Gradients[i] = Gradients[i] * torch.min(torch.ones(1), ClipBound[i] / Gradients_norm)
            Gradients[i] = Gradients[i] + variance * ClipBound[i] * torch.randn(Layers_shape[i])
            Gradients[i] = Gradients[i] * B[i] + M[i]
    if len(ClipBound) == 1:
        for i in range(Layers_num):
            Gradients[i] = Gradients[i] * torch.min(torch.ones(1), ClipBound / Gradients_norm)
            Gradients[i] = Gradients[i] + variance * ClipBound * torch.randn(Layers_shape[i])
            Gradients[i] = Gradients[i] * B[i] + M[i]

    return Gradients

########################## Define training process ########################
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    model.train()
    model.zero_grad()
    output = model(train_data)
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

########################### Define test function #############################
def test(model, device, test_loader):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data = torch.squeeze(data, 1)
            data, target = Variable(data.to(device)), Variable(target.to(device))
            output = model(data)
            test_loss +=  Criteria(output, target) * data.shape[0] # sum up batch loss
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).long().sum()

    test_loss /= test_loader_len
    test_acc = correct.float() / test_loader_len

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_loader_len,
        100. * test_acc))
    return test_loss, test_acc

########################################################################
###################### Load Dataset ##############
# Path
data_root_train = "../data/shakespeare/train/all_data_niid_0_keep_0_train_9.json"
data_root_test = "../data/shakespeare/test/all_data_niid_0_keep_0_test_9.json"
# Load train datasets
train_loader = Datasets.Shakespeare(data_root_train, user_size=args.users_total)
Leaf_split = train_loader.num_samples # Original split provided by LEAF
output_size = len(set(train_loader.targets))
print('Users number is {}, trainning set size is {}, the least samples per user is {}, output size is {} \n split is {}'.
      format(len(Leaf_split), len(train_loader), min(Leaf_split),output_size, Leaf_split ))

# Load test dataset
test_loader = torch.utils.data.DataLoader(Datasets.Shakespeare(data_root_test, user_size=args.users_total),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

###################################################################################
############## Generate model and users ##############
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
########## Assign train dataset to all users/devices ############
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio)
Criteria = nn.CrossEntropyLoss()

###################################################################################
######## Initial model accuracy on test dataset #######
test_loss, test_acc = test(model, device, test_loader)

########## Logging files ############
with open('../results/SHAKESPEARE/AdaClip1_Asyn_TestLoss_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, args.batch_size, args.lr, args.z, args.grad_upper_bound,
                    args.ClipStyle, args.Leaf))
    fl.write(str(test_loss) + '\t')
with open('../results/SHAKESPEARE/AdaClip1_Asyn_TestAcc_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, args.batch_size, args.lr, args.z, args.grad_upper_bound,
                    args.ClipStyle, args.Leaf))
    fl.write(str(test_acc) + '\t')
with open('../results/SHAKESPEARE/AdaClip1_Asyn_TrainLoss_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, args.batch_size, args.lr, args.z, args.grad_upper_bound,
                    args.ClipStyle, args.Leaf))
with open('../results/SHAKESPEARE/AdaClip1_Asyn_Budget_04flat.txt', 'a+') as fl:
    fl.write('\n {} Results \n (UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {}, CS is {}, LF is {})'.
             format(date, Users_num_total, args.user_sel_prob, args.batch_size, args.lr, args.z, args.grad_upper_bound,
                    args.ClipStyle, args.Leaf))

###################################################################################
#################################定义可视化结果####################################
# Define loging dictionary
logs = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'varepsilon': []}

# Define visdom
# Results_testloss = vis.line(np.array([test_loss.numpy()]), [1], win='Test_loss',
#                             opts=dict(title='Test loss on Shakes', legend=['Test loss']))
# Results_testacc = vis.line(np.array([test_acc.numpy()]), [1], win='Test_acc',
#                             opts=dict(title='Test accuracy on Shakes', legend=['Test accuracy']))
# Results_trainloss = vis.line([0.], [1], win='Train_acc',
#                             opts=dict(title='Train loss on Shakes', legend=['Train loss']))
# Results_varepsilon = vis.line([0.], [1], win='varepsilon',
#                             opts=dict(title='Pirvacy budget on Shakes', legend=['Test accuracy']))

###################################################################################
############## Federated learning process ##############

# Obtain information of layers
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
# Set learning rate
lr = args.lr

# Initialize M and S
M, S = InitializeMS(Layers_shape)
# Set clipping bound
Clip_Bound = ClipBound_gerate(args.grad_upper_bound, Layers_nodes, style=args.ClipStyle)
print('Clip bound is {}'.format(Clip_Bound))
# Compute the sampling ratio for moment account
Sampled_ratio = args.user_sel_prob * args.batch_size
delta = 1. / sum(Leaf_split)**(1.1)    # delta in (varepsilon, delta)-DP

# Privac loss of the first iteraiton
varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, 1, delta, 32)
logs['varepsilon'].append(varepsilon)

with open('../results/SHAKESPEARE/AdaClip1_Asyn_Budget_04flat.txt', 'a+') as fl:
    fl.write(str(varepsilon) + '\t')
# vis.line(np.array([varepsilon]), np.array([1]), win=Results_varepsilon, update='replace')

# Define train/test process
for itr in range(1, args.itr_numbers + 1):
    # Select the participants from the total users with the given probability
    Users_Current = np.random.binomial(Users_num_total, args.user_sel_prob, 1).sum()
    if Users_Current == 0:
        Users_Current = 1
    # Compute the standard variance B
    B = ComputeB(S)

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


        # Return gradient and loss on train dataset
        Gradients_Tensor, loss = train(lr, model_round, train_data, train_targets, device, optimizer, gradient=True)
        # Clip and add noise to the gradient
        Gradients_Tensor = Noise_Addition(Layers_num, Layers_shape, Gradients_Tensor, M, B, Clip_Bound)
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

    # Send the updated model to the corresponding participants
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for idx, (params_server, params_client) in enumerate(zip(model.parameters(),worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model

    ################## Update M and S ################
    M, S = UpdateMS(Collect_Gradients, M, S, B, args, Layers_num, Clip_Bound)

    ############ Print train loss #################
    if itr == 1 or itr % args.log_train == 0:
        print('Train Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            itr, itr, args.itr_numbers,
            100. * itr / args.itr_numbers, Loss_train.item()))
        logs['train_loss'].append(Loss_train.item())
        with open('../results/SHAKESPEARE/AdaClip1_Asyn_TrainLoss_04flat.txt', 'a+') as fl:
            fl.write(str(Loss_train.item()) + '\t')
        # if itr == 1:
        #     vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='replace')
        # else:
        #     vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='append')

    ############# Print test loss ############
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)
        varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, itr, delta, 32)
        print('Privacy Budget: {:.6f}; {} -th Iteration'.format(varepsilon, itr))
        logs['varepsilon'].append(varepsilon)
        with open('../results/SHAKESPEARE/AdaClip1_Asyn_TestLoss_04flat.txt', 'a+') as fl:
            fl.write(str(test_loss) + '\t')
        with open('../results/SHAKESPEARE/AdaClip1_Asyn_TestAcc_04flat.txt', 'a+') as fl:
            fl.write(str(test_acc) + '\t')
        with open('../results/SHAKESPEARE/AdaClip1_Asyn_Budget_04flat.txt', 'a+') as fl:
            fl.write(str(varepsilon) + '\t')
        # vis.line(np.array(np.array([test_loss.numpy()])), np.array([itr]), win=Results_testloss, update='append')
        # vis.line(np.array(np.array([test_acc.numpy()])), np.array([itr]), win=Results_testacc, update='append')
        # vis.line(np.array([varepsilon]), np.array([itr]), win=Results_varepsilon, update='append')