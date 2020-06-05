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
import warnings
warnings.filterwarnings("ignore")
date = datetime.now().strftime('%Y-%m-%d %H:%M')
vis = Visdom(env='CELEBA_AdaClip2_Asyn_04_flat')

# Define parameters
class Arguments():
    def __init__(self):
        self.batch_size = 5  # Number of samples used of each user/device at each iteration.
        # If this value is less than 1, then it means the sampling ratio, else it means the mini-batch size
        self.lr = 0.0001  # Learning rate
        self.ClipBound = torch.tensor([0.001])  # clipbound
        self.z = 0.8  # Noise parameter z in Gaussian noise N(0, (zS)^2) where S is sensitivity
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
        self.c = torch.tensor([0.5])  # Split ratio of privacy budget
        self.quatile = 0.5   # Target unclipped gradient

args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

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
############################## Define functions ##################################

################### Define split of users/devices ##############
def Virtual_Users_num(Leaf_split, LEAF=True):
    if LEAF: # Use the original split of users provided by LEAF
        Users_num_total = len(Leaf_split)
        Ratio = Leaf_split
    else: # Self define the split of users
        Users_num_total = args.users_total
        Ratio = [random.randint(1, 10) for _ in range(Users_num_total)]
    return Users_num_total, Ratio

####### Obtain the number and shape of layers of the model ######
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes

################# Initialize all layers as zero ##################
def ZerosGradients(Layers_shape):
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i]))
    return ZeroGradient

############## Define Clipping bound for Flat/Per-Layer ###########
def ClipBound_gerate(Clipbound, Layers_nodes, style="Flat"):
    Layer_nodes = torch.tensor(Layers_nodes).float()
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

#######################Define Clipping for AdaClip2#######################
def TensorClip(Tensor, ClipBound):
    norm = torch.tensor([0.])
    Layers_num = len(Tensor)
    Tuple_num = torch.ones(Layers_num)
    for i in range(Layers_num):# For each layer
        #Count the number of its nodes and compute the its norm
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

#################### Define clipping ######################

def AddNoise_to_Model(Tensor, Clipbound, args):
    std =  args.z * Clipbound.float().norm() *torch.sqrt(1/(1-args.c))/(args.users_total * args.user_sel_prob)
    for i in range(len(Tensor)):
        Tensor[i] = Tensor[i] + std*torch.randn(Tensor[i].shape)
    return Tensor

################### Define adding noise ####################
def UpdateClipBound(args, Clipbound, lr, Indicator, style="Lin"):
    std = args.z * torch.sqrt(1/args.c)/(args.users_total * args.user_sel_prob)
    Indicator = Indicator + std*torch.randn_like(Indicator)
    if style == "Lin":
        Clipbound = Clipbound - lr * (Indicator - args.quatile)
    if style == "Geo":
        Clipbound = Clipbound * torch.exp(-lr * (Indicator - args.quatile))
    return Clipbound

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

########################### Define test function ###################
def test(model, device, test_loader):
    model.eval()
    test_loader_len = len(test_loader.dataset)
    test_loss = 0
    correct = 0
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

####################################################################
######################Load Dataset##############
#Path
data_root_train = "../data/CELEBA/raw_data/train_data.json"
data_root_test = "../data/CELEBA/raw_data/test_data.json"
IMAGES_DIR = "../data/CELEBA/raw_data/img_align_celeba/"
#Load train datasets
train_loader = Datasets.celeba(data_root_train, IMAGES_DIR, args.users_total)#训练集载入
Leaf_split = train_loader.num_samples#LEAF提供的用户数量和训练集上的数据切分
#Load train datasets
test_loader = torch.utils.data.DataLoader(Datasets.celeba(data_root_test, IMAGES_DIR, args.users_total),
    batch_size=args.test_batch_size, shuffle=True, drop_last=True, **kwargs)

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
print('THe number of total users: {}, the least samples of users is {}'.format(Users_num_total, min(Leaf_split)))

################################ Define logging files #################################
with open('../results/CELEBA/AdaClip2_Asyn_08flat_TestLoss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', BZ is {}, CB is {}, lr is {}, z is {}, {}, c is {}'
             .format(args.batch_size, args.ClipBound, args.lr, args.z, args.ClipStyle, args.c))
    fl.write(', quantile is {}, total iteration is {}, log_test is {}, log_train is {}\n'
             .format(args.quatile, args.itr_numbers, args.log_test, args.log_train))

with open('../results/CELEBA/AdaClip2_Asyn_08flat_TestAcc.txt', 'a+') as fl:
    fl.write('\n %' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', BZ is {}, CB is {}, lr is {}, z is {}, {}, c is {}'.format(args.batch_size,args.ClipBound, args.lr, args.z, args.ClipStyle, args.c))
    fl.write(', quantile is {}, total iteration is {}, log_test is {}, log_train is {}\n'.format(args.quatile, args.itr_numbers, args.log_test, args.log_train))

with open('../results/CELEBA/AdaClip2_Asyn_08flat_TrainLoss.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', BZ is {}, CB is {}, lr is {}, z is {}, {}, c is {}'
            .format(args.batch_size, args.ClipBound, args.lr,args.z, args.ClipStyle, args.c))
    fl.write(', quantile is {}, total iteration is {}, log_test is {}, log_train is {}\n'
             .format(args.quatile,args.itr_numbers,args.log_test,args.log_train))

with open('../results/CELEBA/AdaClip2_Asyn_08flat_Budget.txt', 'a+') as fl:
    fl.write('\n' + date + ' Results (Total users number is ' + str(args.users_total))
    fl.write(', BZ is {}, CB is {}, lr is {}, z is {}, {}, c is {}'
             .format(args.batch_size, args.ClipBound, args.lr, args.z, args.ClipStyle, args.c))
    fl.write(', quantile is {}, total iteration is {}, log_test is {}, log_train is {}\n'
             .format(args.quatile, args.itr_numbers, args.log_test, args.log_train))
logs = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'varepsilon': []}

###################################################################################
########## Assign train dataset to all users/devices ############
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio)
# Criteria = nn.CrossEntropyLoss()

######## Initial model accuracy on test dataset #######
test_loss, test_acc = test(model, device, test_loader)
logs['test_loss'].append(test_loss)
logs['test_acc'].append(test_acc)
###################################################################################
##############Federated learning process##############
#Define visdom

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
# Generate clipping bound
Clip_Bound = ClipBound_gerate(args.ClipBound, Layers_nodes, style=args.ClipStyle)

# Compute the sampling ratio for moment account
Sampled_ratio = args.user_sel_prob
delta = 1. / Users_num_total ** 1.1 # delta in (varepsilon, delta)-DP

#Privac loss of the first iteraiton
varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, 1, delta, 32)
logs['varepsilon'].append(varepsilon)

# Define train/test process
for itr in range(1, args.itr_numbers + 1):
    #Select the participants from the total users with the given probability
    Users_Current = np.random.binomial(Users_num_total, args.user_sel_prob, 1).sum()

    # Load samples from the participants with the given probability or mini-batch size args.batch_size
    federated_train_loader = sy.FederatedDataLoader(Federate_Dataset, batch_size=args.batch_size, shuffle=True,
                                                    worker_num=Users_Current, batch_num=args.batchs_round, **kwargs)
    workers_list = federated_train_loader.workers # List of participants at the current iteration

    # Next two lines are only necessary for synchronous aggregation
    # for idx in range(len(workers_list)):
    #     models[workers_list[idx]] = model

    # Initialize the same model-structure tensor with zero elements
    Collect_Models = ZerosGradients(Layers_shape)
    Collect_Indicator = torch.zeros_like(Clip_Bound)
    Loss_train = torch.tensor(0.)
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()
        # optimizer = optims[data.location.id]
        # Return model parameters and loss on train dataset
        ModelVary_Tensor, loss = train(lr, model_round, train_data, train_targets, device, optimizer, gradient=False)
        Loss_train += loss
        # Model clipping
        ModelVary_Tensor, Indicator = TensorClip(ModelVary_Tensor, Clip_Bound)
        # Accumulation model parameters for participants
        for i in range(Layers_num):
            Collect_Models[i] = Collect_Models[i] + ModelVary_Tensor[i]
        #  Accumulation indicators for participants
        for i in range(len(Clip_Bound)):
            Collect_Indicator[i] = Collect_Indicator[i] + Indicator[i]
    # Average model paremeters
    for i in range(Layers_num):
        Collect_Models[i] = Collect_Models[i]/ (idx_outer+1)
    Collect_Models = AddNoise_to_Model(Collect_Models, Clip_Bound, args)  # Noise injectting
    # Average indicators
    Collect_Indicator = Collect_Indicator/ (idx_outer+1)
    # Update clipping bound
    Clip_Bound = UpdateClipBound(args, Clip_Bound, lr, Collect_Indicator, style="Geo")

    # Update the global model using the average model parameters
    for idx_para, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(1, Collect_Models[idx_para])

    # Average train loss
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
        if itr == 1:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='replace')
        else:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='append')


        ############ Print the test loss and accuracy ###########
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)
        varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, itr, delta, 32)
        print('Privacy Budget: {:.6f}; {} -th Iteration'.format(varepsilon, itr))
        logs['varepsilon'].append(varepsilon)
        vis.line(np.array(np.array([test_loss])), np.array([itr]), win=Results_testloss, update='append')
        vis.line(np.array(np.array([test_acc])), np.array([itr]), win=Results_testacc, update='append')


with open('../results/CELEBA/AdaClip2_Asyn_08flat_TrainLoss.txt', 'a+') as fl:
    fl.write('train_loss: ' + str(logs['train_loss']))
with open('../results/CELEBA/AdaClip2_Asyn_08flat_TestLoss.txt', 'a+') as fl:
    fl.write('test_loss: ' + str(logs['test_loss']))
with open('../results/CELEBA/AdaClip2_Asyn_08flat_TestAcc.txt', 'a+') as fl:
    fl.write('test_acc: '+str(logs['test_acc']))
with open('../results/CELEBA/AdaClip2_Asyn_08flat_Budget.txt', 'a+') as fl:
    fl.write('Total privacy loss: ' + str(logs['varepsilon']))