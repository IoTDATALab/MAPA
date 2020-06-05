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
import ComputePrivacy as Privacy#导入自定义计算隐私预算函数
import logging
import Datasets

logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')
vis = Visdom(env='SENT140_AdaClip2_ASyn')

#定义参量
class Arguments():
    def __init__(self):
        self.batch_size = 1
        self.test_batch_size = 1
        self.epochs = 2000
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_train = 100
        self.log_test = 100
        self.save_model = False
        self.users_total = 1000#虚拟用户总量
        self.users_round = 10#每回合参与用户
        self.user_sel_prob = 0.01 #每个用户抽取概率
        self.batchs_round = 1#每回合单个用户使用的批数量
        self.grad_upper_bound = torch.tensor([0.1]) #预估初始梯度上界
        self.variance = 0.01#噪声的标准差
        self.itr_numbers = 5000#迭代总数
        self.c = torch.tensor([0.5])#隐私在模型更新和clip bound调节的分配比例
        self.z = 0.3#信噪比
        self.quatile = 0.5#模型调节的分位数


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
###############################函数定义############################################

################################定义虚拟用户数量#######################
def Virtual_Users_num(Leaf_split, LEAF=True):
    Users_num_total = 0
    if LEAF:
        Users_num_total = len(Leaf_split)
        Ratio = Leaf_split
    else:
        Users_num_total = args.users_total
        #生成用户数据切分比例
        Ratio = [random.randint(1, 10) for _ in range(Users_num_total)]
    return Users_num_total, Ratio

##################################获取模型层数和各层的形状#############
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes

##################################设置各层的梯度为0#####################
def ZerosGradients(Layers_shape):
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i]))
    return ZeroGradient

#############################定义Clipping bound生成#####################
def ClipBound_gerate(Clipbound, Layers_nodes, style="Flat"):
    Layer_nodes = torch.tensor(Layers_nodes).float()
    if style == "Flat":
        ClipBound = Clipbound
    if style == "Per-Layer":
        ClipBound = Layer_nodes/Layer_nodes.norm() * Clipbound
    return ClipBound

################################定义剪裁#################################
def TensorClip(Tensor, ClipBound):
    norm = torch.tensor([0.])
    Layers_num = len(Tensor)
    Tuple_num = torch.ones(Layers_num)
    for i in range(Layers_num):#统计所有层拉平后的范数和各层的神经元个数
        norm = norm + Tensor[i].float().norm()**2
        Tuple_num[i] = Tensor[i].numel()
    norm = norm.sqrt()
    if len(ClipBound) == 1:
        Indicator = 1*(norm<=ClipBound[0])
        for i in range(Layers_num):
            Tensor[i] = Tensor[i]*torch.min(torch.ones(1), ClipBound[0]/norm)
    if len(ClipBound) > 1:
        Indicator = torch.zeros(Layers_num)
        # ClipBound_layers = Tuple_num.float()/Tuple_num.norm()*ClipBound#按各层的神经元数目分割ClipBound
        for i in range(Layers_num):
            norm_layer = Tensor[i].float().norm()
            Indicator[i] = 1*(norm_layer<=ClipBound[i])
            Tensor[i] = Tensor[i]*torch.min(torch.ones(1), ClipBound[i]/norm_layer)
    return Tensor, Indicator

####################################定义噪声添加#########################
#
def AddNoise_to_Model(Tensor, Clipbound, args):
    std =  args.z * Clipbound.float().norm() *torch.sqrt(1/(1-args.c))/(args.users_total * args.user_sel_prob)
    for i in range(len(Tensor)):
        Tensor[i] = Tensor[i] + std*torch.randn(Tensor[i].shape)
    return Tensor

#################################定义更新clip bound######################
def UpdateClipBound(args, Clipbound, lr, Indicator, style="Lin"):
    std = args.z * torch.sqrt(1/args.c)/(args.users_total * args.user_sel_prob)
    Indicator = Indicator + std * torch.randn_like(Indicator)
    if style == "Lin":
        Clipbound = Clipbound - lr * (Indicator - args.quatile)
    if style == "Geo":
        Clipbound = Clipbound * torch.exp(-lr * (Indicator - args.quatile))
    return Clipbound

##########################定义训练过程，返回梯度########################
def train(learning_rate, model, train_data, train_target, idx_unsort, gradient=True):
    model.train()
    model.zero_grad()
    output = model(train_data)
    output = output.index_select(0,idx_unsort)#对之前的排序进行还原，以便能和target对应
    loss = Criteria(output, train_target)
    loss.backward()
    Gradients_Tensor = []
    if gradient == False:
        for params in model.parameters():
            Gradients_Tensor.append(-learning_rate*params.grad.data)#返回-lr*grad
    if gradient == True:
        for params in model.parameters():
            Gradients_Tensor.append(params.grad.data)#把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss

############################定义测试函数################################
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
###############################数据载入############################################
#载入训练集和测试集
vocab_root = '../data/sentiment140/embs.json'
data_root_train = "../data/sentiment140/train/all_data_niid_0_keep_0_train_9.json"
data_root_test = "../data/sentiment140/test/all_data_niid_0_keep_0_test_9.json"
#训练集
train_loader = Datasets.sentiment140(data_root_train, vocab_root, user_size=args.users_total)#训练集载入
Leaf_split = train_loader.num_samples#LEAF提供的用户数量和训练集上的数据切分
output_size = len(set(train_loader.targets))
print('Users number is {}, trainning set size is {}, the least samples per user is {}, output size is {} \n split is {}'.
      format(len(Leaf_split), len(train_loader), min(Leaf_split),output_size, Leaf_split ))
batch_size = min(min(Leaf_split), args.batch_size)

#测试集
test_loader = torch.utils.data.DataLoader(Datasets.sentiment140(data_root_test,vocab_root, user_size=args.users_total),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

###################################################################################
##################################模型和用户生成###################################
model = Net().to(device)
workers = []
models = {}
optims = {}
Users_num_total, Ratio = Virtual_Users_num(Leaf_split, LEAF=True)
for i in range(1, Users_num_total+1):
    exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i,i))
    exec('models["user{}"] = model.copy()'.format(i))
    exec('optims["user{}"] = optim.SGD(params=models["user{}"].parameters(), lr=args.lr)'.format(i,i))
    exec('workers.append(user{})'.format(i))#列表形式存储用户
    # exec('workers["user{}"] = user{}'.format(i,i))#字典形式存储用户
optim_sever = optim.SGD(params=model.parameters(),lr=args.lr)#定义服务器优化器

###################################################################################
###############################联邦数据集生成######################################
Federate_Dataset = Datasets.dataset_federate_noniid(train_loader, workers, Ratio=Ratio)
Criteria = nn.CrossEntropyLoss()

###################################################################################
########################生成文件用于记录实验结果###################################
test_loss, test_acc = test(model, device, test_loader) # 初始模型的预测精度
with open('../实验结果/SENT140/AdaClip2_Asyn_TestLoss.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Per-Layer, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))
    fl.write(str(test_loss.item()) + '\t')
with open('../实验结果/SENT140/AdaClip2_Asyn_TestAcc.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Per-Layer, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))
    fl.write(str(test_acc.item()) + '\t')
with open('../实验结果/SENT140/AdaClip2_Asyn_TrainLoss.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Per-Layer, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))
with open('../实验结果/SENT140/AdaClip2_Asyn_Budget.txt', 'a+') as fl:
    fl.write('\n {} Results \n (Per-Layer, UN is {}, SP is {}, BZ is {}, LR is {}, SNR is {}, CB is {})'.
             format(date, args.users_total, args.user_sel_prob, batch_size, args.lr, args.z, args.grad_upper_bound))

###################################################################################
#################################定义可视化结果####################################
#定义记录字典
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
#################################联邦学习过程######################################
#获取模型层数和各层形状
Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
#设置学习率
lr = args.lr
#设置clipping bound
Clip_Bound = ClipBound_gerate(args.grad_upper_bound, Layers_nodes, style="Per-Layer")
#设置每回合用户的抽样个数
Sampled_num = args.users_total * args.user_sel_prob * batch_size * args.batchs_round
Sampled_ratio = Sampled_num / sum(Leaf_split)#抽样比例，用于计算隐私预算
delta = 1. / sum(Leaf_split)**(1.1)#(varepsilon, delta)中delta

#初始时刻的隐私预算
varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, 1, delta, 32)
logs['varepsilon'].append(varepsilon)
with open('../实验结果/SENT140/AdaClip2_Asyn_Budget.txt', 'a+') as fl:
    fl.write(str(varepsilon) + '\t')
vis.line(np.array([varepsilon]), np.array([1]), win=Results_varepsilon, update='replace')

#定义训练/测试过程
for itr in range(1, args.itr_numbers + 1):
    #按概率0.1生成当前回合用户数量
    Users_Current = np.random.binomial(Users_num_total, args.user_sel_prob, 1).sum()
    if Users_Current == 0:
        Users_Current = 1

    #按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
    #为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
    federated_train_loader = sy.FederatedDataLoader(Federate_Dataset, batch_size=batch_size, shuffle=True,
                                                    worker_num=Users_Current, batch_num=args.batchs_round, **kwargs)
    workers_list = federated_train_loader.workers#当前回合抽取的用户列表

    #同步更新: 每回合迭代开始前，把当前服务器模型发送给对应的用户，需要下面2行代码
    #异步设置，注释掉这两行
    # for idx in range(len(workers_list)):
    #     models[workers_list[idx]] = model

    # 生成与模型梯度结构相同的元素=0的列表
    Collect_Models = ZerosGradients(Layers_shape)
    Collect_Indicator = torch.zeros_like(Clip_Bound)
    Loss_train = torch.tensor(0.)
    for idx_outer, (train_data, train_targets) in enumerate(federated_train_loader):
        model_round = models[train_data.location.id]
        optimizer = optims[train_data.location.id]
        train_data, train_targets = train_data.to(device), train_targets.to(device)
        train_data, train_targets = train_data.get(), train_targets.get()

        train_targets, length = torch.split(train_targets, 1, dim=1)#分割target 和 length
        length = torch.squeeze(length, dim=1)
        train_targets = torch.squeeze(train_targets, dim=1)
        train_data = Variable(train_data)

        # 对data按照句子的长短进行排序
        _, idx_sort = torch.sort(length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        train_data = train_data.index_select(0, idx_sort)
        length = length.index_select(0, idx_sort)

        train_data = torch.nn.utils.rnn.pack_padded_sequence(train_data, length, batch_first=True)  # pack

        # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad
        ModelVary_Tensor, loss = train(lr, model_round, train_data, train_targets, idx_unsort, gradient=False)
        Loss_train += loss
        #模型裁剪，如果ClipBound是一维，则是Flat Clipping；否则是Per - layer Clipping
        ModelVary_Tensor, Indicator = TensorClip(ModelVary_Tensor, Clip_Bound)
        # 累加本回合不同用户发送的噪声模型变化
        for i in range(Layers_num):
            Collect_Models[i] = Collect_Models[i] + ModelVary_Tensor[i]
        # 累加本回合示性函数和
        for i in range(len(Clip_Bound)):
            Collect_Indicator[i] = Collect_Indicator[i] + Indicator[i]
    #平均化噪声梯度
    for i in range(Layers_num):
        Collect_Models[i] = Collect_Models[i]/ (idx_outer + 1)
    Collect_Models = AddNoise_to_Model(Collect_Models, Clip_Bound, args)  # 模型添加噪声
    #平均化示性函数
    Collect_Indicator = Collect_Indicator/ (idx_outer + 1)
    #更新ClipBound
    Clip_Bound = UpdateClipBound(args, Clip_Bound, lr, Collect_Indicator, style="Geo")

    #利用平均化噪声模型改变量更新服务器模型并且把更新后的模型发送给对应学习者
    for idx_para, params_sever in enumerate(model.parameters()):
        params_sever.data.add_(1, Collect_Models[idx_para])

    #平均训练损失
    Loss_train /= (idx_outer + 1)
    #同步更新不需要下面代码；异步更新需要下段代码
    for worker_idx in range(len(workers_list)):
        worker_model = models[workers_list[worker_idx]]
        for _, (params_server, params_client) in enumerate(zip(model.parameters(),worker_model.parameters())):
            params_client.data = params_server.data
        models[workers_list[worker_idx]] = worker_model###添加把更新后的模型返回给用户

        ############间隔给定迭代次数打印损失#################
    if itr == 1 or itr % args.log_train == 0:
        print('Train Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            itr, itr, args.itr_numbers,
            100. * itr / args.itr_numbers, Loss_train.item()))
        logs['train_loss'].append(Loss_train.item())
        with open('../实验结果/SENT140/AdaClip2_Asyn_TrainLoss.txt', 'a+') as fl:
            fl.write(str(Loss_train.item()) + '\t')
        if itr == 1:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='replace')
        else:
            vis.line(np.array([Loss_train.data.numpy()]), np.array([itr]), win=Results_trainloss, update='append')

        #############间隔给定迭代次数打印预测精度############
    if itr % args.log_test == 0:
        test_loss, test_acc = test(model, device, test_loader)
        logs['test_loss'].append(test_loss)
        logs['test_acc'].append(test_acc)
        varepsilon = Privacy.ComputePrivacy(Sampled_ratio, args.z, itr, delta, 32)
        print('Privacy Budget: {:.6f}; {} -th Iteration'.format(varepsilon, itr))
        logs['varepsilon'].append(varepsilon)
        with open('../实验结果/SENT140/AdaClip2_Asyn_TestLoss.txt', 'a+') as fl:
            fl.write(str(test_loss.item()) + '\t')
        with open('../实验结果/SENT140/AdaClip2_Asyn_TestAcc.txt', 'a+') as fl:
            fl.write(str(test_acc.item()) + '\t')
        with open('../实验结果/SENT140/AdaClip2_Asyn_Budget.txt', 'a+') as fl:
            fl.write(str(varepsilon) + '\t')
        vis.line(np.array(np.array([test_loss.numpy()])), np.array([itr]), win=Results_testloss, update='append')
        vis.line(np.array(np.array([test_acc.numpy()])), np.array([itr]), win=Results_testacc, update='append')
        vis.line(np.array([varepsilon]), np.array([itr]), win=Results_varepsilon, update='append')