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
import ComputePrivacy as Privacy # Import self definition function to compute the privacy loss
import logging
import Datasets # Import self definition function to load the federated datasets
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')
vis = Visdom(env='CELEBA_AdaClip1_Asyn_04_flat')

# Define parameters
class Arguments():
    def __init__(self):
        self.batch_size = 5  # Number of samples used of each user/device at each iteration.
        # If this value is less than 1, then it means the sampling ratio, else it means the mini-batch size
        self.lr = 1  # Learning rate
        self.ClipBound = torch.tensor([0.1])  # clipbound
        self.z = 0.4  # Noise parameter z in Gaussian noise N(0, (zS)^2) where S is sensitivity
        self.users_total = 800  # Total number of users/devices
        self.user_sel_prob = 0.02  # Probability for sampling users/devices at each iteration
        self.itr_numbers = 6000  # Number of total iterations

        self.test_batch_size = 128  # Number of test mini-batch size
        self.log_train = 100  # Logging interval for printing the training loss
        self.log_test = 100  # Logging interval for printing the test accuracy
        self.save_model = False
        self.batchs_round = 1  # Number of mini-batchs of each selected user in each iteration
        self.no_cuda = True
        self.seed = 1
        self.ClipStyle = 'Flat'  # Clipping method, including Flat and Per-Layer
        self.beta_1 = 0.99  # Averaging speed parameter for the mean of gradients
        self.beta_2 = 0.9  # Averaging speed parameter for the variance of gradients
        self.h_1 = torch.tensor(10 ** (-12))  # Parameter 1 for estimating stand variance
        self.h_2 = torch.tensor(10 ** (-10))  # Parameter 2 for estimating stand variance

args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 4, 1)
        #self.conv2 = nn.Conv2d(10, 20, 4, 1)
        self.fc1 = nn.Linear(40*40*10, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 40*40*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

###################################################################################
############################### Define functions ##################################

################### Define split of users/devices ##############
def Virtual_Users_num(Leaf_split, LEAF=True):
    if LEAF: # Use the original split of users provided by LEAF
        Users_num_total = len(Leaf_split)
        Ratio = Leaf_split
    else: # Self define the split of users
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
    if len(Clip_Bound) == 1:
        for i in range(Layers_num):
            M[i] = args.beta_1 * M[i] + (1 - args.beta_1) * Gradients[i]
            temp = torch.min(torch.max(((Gradients[i] - M[i])**2 - B[i]**2 * (args.z*Clip_Bound[0])**2), args.h_1.float()), args.h_2.float())
            S[i] = torch.sqrt(args.beta_2 * S[i]**2 + (1 - args.beta_2)*temp)
    elif len(Clip_Bound) > 1:
        for i in range(Layers_num):
            M[i] = args.beta_1 * M[i] + (1 - args.beta_1) * Gradients[i]
            temp = torch.min(torch.max(((Gradients[i] - M[i])**2 - B[i]**2 * (args.z*Clip_Bound[i])**2), args.h_1.float()), args.h_2.float())
            S[i] = torch.sqrt(args.beta_2 * S[i]**2 + (1 - args.beta_2)*temp)
    return M, S

################## Define Clipping bound for Flat/Per-Layer ###############
def ClipBound_gerate(Clipbound, Layers_nodes, style="Flat"):
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        Layer_nodes = torch.tensor(Layers_nodes).float()
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

#############Define clipping and adding noise#################
def Noise_Addition(Layers_num, Layers_shape, Gradients, M, B, ClipBound):
    variance = args.z
    if len(ClipBound)==1:
        Gradients_norm = torch.tensor([0.])
        for i in range(Layers_num):
            Gradients[i] = (Gradients[i] - M[i])/B[i]
            Gradients_norm = Gradients_norm + Gradients[i].norm()**2
        Gradients_norm = torch.sqrt(Gradients_norm)
        for i in range(Layers_num):
            if ClipBound < Gradients_norm:
                Gradients[i] = ClipBound * Gradients[i] / Gradients_norm
            Gradients[i] = Gradients[i] + variance * ClipBound * torch.randn(Layers_shape[i])
            Gradients[i] = Gradients[i]*B[i] + M[i]
    elif len(ClipBound)>=1:
        Gradients_norm = torch.zeros([6])
        for i in range(Layers_num):
            Gradients[i] = (Gradients[i] - M[i])/B[i]
            Gradients_norm[i] = Gradients[i].norm()
        for i in range(Layers_num):
            if ClipBound[i] < Gradients_norm[i]:
                Gradients[i] = ClipBound[i] * Gradients[i] / Gradients_norm[i]
            Gradients[i] = Gradients[i] + variance * ClipBound[i] * torch.randn(Layers_shape[i])
            Gradients[i] = Gradients[i]*B[i] + M[i]
    return Gradients

##########################Define training process########################
def train(learning_rate, model, train_data, train_target, device, optimizer, gradient=True):
    # If gradient=Ture, then return gradient, else return model parameters
    model.train()
    model.zero_grad()
    output = model(train_data.float())
    loss = F.nll_loss(output, train_target.long())
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
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss +=  F.nll_loss(output, target.long(), reduction='sum').item() # sum up batch loss
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
#Path
data_root_train = "../data/CELEBA/raw_data/train_data.json"
data_root_test = "../data/CELEBA/raw_data/test_data.json"
IMAGES_DIR = "../data/CELEBA/raw_data/img_align_celeba/"
#Load train datasets
train_loader = Datasets.celeba(data_root_train, IMAGES_DIR,args.users_total)
Leaf_split = train_loader.num_samples # original split provided by LEAF
# Load train datasets
test_loader = torch.utils.data.DataLoader(Datasets.celeba(data_root_test, IMAGES_DIR,args.users_total),
    batch_size=args.test_batch_size, shuffle=True, drop_last=True, num_workers=0, **kwargs)

######################################################################
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
optim_sever = optim.SGD(params=model.parameters(),lr=args.lr) # Define global model
Leaf_split = np.round(len(train_loader)*np.array(Ratio)/sum(Ratio)).tolist()
print('THe number of total users: {}, the least samples of users is {}'.format(Users_num_total, min(Leaf_split)))

###################################################################################
##########Assign train dataset to all users/devices############
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio)
# Criteria = nn.CrossEntropyLoss()

########Initial model accuracy on test dataset#######
test_loss, test_acc = test(model, device, test_loader)


######################################################################
##############Federated learning process##############
#Define visdom
logs = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'varepsilon': []}
logs['test_loss'].append(test_loss)
logs['test_acc'].append(test_acc)
Results_testloss = vis.line(np.array([test_loss]), [1], win='Test_loss',
                            opts=dict(title='Test loss on Sent140', legend=['Test loss']))
Results_testacc = vis.line(np.array(np.array([test_acc])), [1], win='Test_acc',
                            opts=dict(title='Test accuracy on Sent140', legend=['Test accuracy']))
Results_trainloss = vis.line([0.], [1], win='Train_acc',
                            opts=dict(title='Train loss on Sent140', legend=['Train loss']))

# Obtain information of layers
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
# Set learning rate
lr = args.lr
# Initialie M and S
M, S = InitializeMS(Layers_shape)
# Set clipping bound
Clip_Bound = ClipBound_gerate(args.ClipBound, Layers_nodes, style=args.ClipStyle)
# Compute the sampling ratio for moment account
Sampled_ratio = args.user_sel_prob * args.batch_size
delta = 1. / sum(Leaf_split) ** 1.1     # delta in (varepsilon, delta)-DP

#Privac loss of the first iteraiton
varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, 1, delta, 32)
logs['varepsilon'].append(varepsilon)


# Define train/test process
for itr in range(1, args.itr_numbers + 1):
    #Select the participants from the total users with the given probability
    Users_Current = np.random.binomial(Users_num_total, args.user_sel_prob, 1).sum()
    # Compute the standard variance B
    B = ComputeB(S)

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

    # Send the updated model to the corresponding participants, only for asynchronous aggregation
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for _, (params_server, params_client) in enumerate(zip(model.parameters(),worker_model.parameters())):
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
        if itr == 1:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='replace')
        else:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='append')


    ############ Print test loss and accuracy #################
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)
        varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, itr, delta, 32)
        print('Privacy Budget: {:.6f}; {} -th Iteration'.format(varepsilon, itr))
        logs['varepsilon'].append(varepsilon)
        vis.line(np.array(np.array([test_loss])), np.array([itr]), win=Results_testloss, update='append')
        vis.line(np.array(np.array([test_acc])), np.array([itr]), win=Results_testacc, update='append')

with open('../results/CELEBA/AdaClip1_Asyn_04flat_TestLoss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', batch_size is {}, learning rate is {}, ClipBound is {}, z is {}, {},'
             .format(args.batch_size, args.lr, args.ClipBound, args.z, args.ClipStyle))
    fl.write(', beta1 is {}, beta2 is {}, h1 is {}, h2 is {}, itr_number is {}, log_test is {}, log_train is {} \n'
             .format(args.beta_1, args.beta_2, args.h_1, args.h_2, args.itr_numbers, args.log_test, args.log_train))
with open('../results/CELEBA/AdaClip1_Asyn_04flat_Budget.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', batch_size is {}, learning rate is {}, ClipBound is {}, z is {}, {},'
             .format(args.batch_size, args.lr, args.ClipBound, args.z, args.ClipStyle))
    fl.write(', beta1 is {}, beta2 is {}, h1 is {}, h2 is {}, itr_number is {}, log_test is {}, log_train is {} \n'
             .format(args.beta_1, args.beta_2, args.h_1, args.h_2, args.itr_numbers, args.log_test, args.log_train))

with open('../results/CELEBA/AdaClip1_Asyn_04flat_TestAcc.txt', 'a+') as fl:
    fl.write('\n %' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', batch_size is {}, learning rate is {}, ClipBound is {}, z is {}, {},'
             .format(args.batch_size, args.lr, args.ClipBound, args.z, args.ClipStyle))
    fl.write(', beta1 is {}, beta2 is {}, h1 is {}, h2 is {}, itr_number is {}, log_test is {}, log_train is {} \n'
             .format(args.beta_1, args.beta_2, args.h_1, args.h_2, args.itr_numbers, args.log_test, args.log_train))

with open('../results/CELEBA/AdaClip1_Asyn_04flat_TrainLoss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', batch_size is {}, learning rate is {}, ClipBound is {}, z is {}, {},'
             .format(args.batch_size, args.lr, args.ClipBound, args.z, args.ClipStyle))
    fl.write(', beta1 is {}, beta2 is {}, h1 is {}, h2 is {}, itr_number is {}, log_test is {}, log_train is {} \n'
             .format(args.beta_1, args.beta_2, args.h_1, args.h_2, args.itr_numbers, args.log_test, args.log_train))

with open('../results/CELEBA/AdaClip1_Asyn_04flat_TrainLoss.txt', 'a+') as fl:
    fl.write('train_loss: ' + str(logs['train_loss']))
with open('../results/CELEBA/AdaClip1_Asyn_04flat_Budget.txt', 'a+') as fl:
    fl.write('Total privacy loss: ' + str(logs['varepsilon']))
with open('../results/CELEBA/AdaClip1_Asyn_04flat_TestLoss.txt', 'a+') as fl:
    fl.write('test_loss: ' + str(logs['test_loss']))
with open('../results/CELEBA/AdaClip1_Asyn_04flat_TestAcc.txt', 'a+') as fl:
    fl.write('test_acc: ' + str(logs['test_acc']))
