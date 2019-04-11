import numpy as np
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import visdom
import math
import matplotlib.pyplot as plt 
import scipy
import h5py
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

#viz = visdom.Visdom()

#parameters
batch_size=1

# Pretrained weight 
w1 = []
w2 = []
w3 = []
w4 = []
w5 = []
w6 = []
w7 = []
w8 = []
b1 = []
b2 = []
b3 = []
b4 = []
b5 = []
b6 = []
b7 = []
b8 = []

class dnn(nn.Module):
    
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
    # Layers 
    def __init__(self, G_in, w1):
        super(dnn, self).__init__()
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, G_in)

    # Deep neural network [you are passing data layer-to-layer]
    def forward(self, x):
        y = F.relu(self.fc1(x))
        x = F.relu(self.fc2(y))
        ww1 = self.fc1.weight
        ww2 = self.fc2.weight
        bb1 = self.fc1.bias
        bb2 = self.fc2.bias
        return x,y,ww1,ww2,bb1,bb2

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


# Class for load the data into system
class speech_data(Dataset):
    
    def __init__(self, folder_path):
        self.path = folder_path
        self.files = listdir(folder_path)
        self.length = len(self.files)
        
    def __getitem__(self, index):
        d = loadmat(join(self.path, self.files[int(index)]))
        return np.array(d['Clean_cent']), np.array(d['Feat'])
    
    def __len__(self):
        return self.length
        
# Path where you want to store your results
mainfolder = "/home/harshit/Desktop/pytorch/VCC/DNN"

# Training Data path
traindata = speech_data(folder_path="/home/harshit/Desktop/DNN_VC/WHSP2SPCH_MCEP/batches/Training_complementary_feats")
train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=2)

# Path for validation data
valdata = speech_data(folder_path="/home/harshit/Desktop/DNN_VC/WHSP2SPCH_MCEP/batches/Validation_complementary_feats")
val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=2)

# Loss function
mmse_loss = nn.MSELoss()

# initialize the Deep neural network
net = dnn(40, 100)#.cuda()

# Optimizer [Adam]
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
layer_1_output = []
    

# Function for training the data
def training(data_loader, n_epochs):
    net.train()
    for en, (b, a) in enumerate(data_loader):
        a = Variable(a.squeeze(0))#.cuda()
        t1,t2 = a.shape
        noise = Variable(torch.randn(t1, t2))
        y = a + noise
        optimizer.zero_grad()
        out, temp, ww1, ww2, bb1, bb2 = net(y)
        layer_1_output.append(temp.detach().numpy())
        loss = mmse_loss(out, a)*5
        loss.backward()
        optimizer.step()
        print ("[Epoch: %d] [Iter: %d/%d] [Loss: %f]" % (n_epochs, en, len(data_loader), loss.cpu().data.numpy()))

# Function that validate our model after every epoch 
def validating(data_loader, epoch):
    net.eval()
    running_loss = 0
    for en, (b, a) in enumerate(data_loader):
        a = Variable(a.squeeze(0))#.cuda()
        out, temp, ww1, ww2, bb1, bb2 = net(a)
        loss = mmse_loss(out, a)*5
        running_loss += loss.item()
        if epoch==2 and not w1:
            w1.append(ww1.detach().numpy())
            w8.append(ww2.detach().numpy())
            b1.append(bb1.detach().numpy())
            b8.append(bb2.detach().numpy())
    return running_loss/(en+1)

# For traning, it is true. For testing, make it 'False'
isTrain = True
if isTrain:
    epoch = 2
    for ep in range(epoch):
        training(train_dataloader, ep+1)
        gl = validating(val_dataloader, ep+1)
        print("loss: " + str(gl))

print("1st Layer Optimization finished!")
# Array for layer 2 feeding
input_layer_2 = np.array(layer_1_output)
##############################################################################

class dnn1(nn.Module):
    
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        
    def __init__(self, G_in, w1):
        super(dnn1, self).__init__()    
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, G_in)
    
    # Deep neural network [you are passing data layer-to-layer]
    def forward(self, x):    
        y = F.relu(self.fc1(x))
        x = F.relu(self.fc2(y))
        ww1 = self.fc1.weight
        ww2 = self.fc2.weight
        bb1 = self.fc1.bias
        bb2 = self.fc2.bias
        return x, y, ww1, ww2, bb1, bb2

# initialize the Deep neural network
net1 = dnn1(100, 40)#.cuda()

# Optimizer [Adam]
optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.01)

# Contractive Autoencoder Loss
def constractive_loss(y_pred, y_true, h, W):
    lamda = 1e-4
    mse = mmse_loss(y_pred, y_true)
    dh = h*(1-h)
    w_sum = torch.sum(Variable(W)**2, dim=1)
    w_sum = w_sum.unsqueeze(1)  # Shape: N_hissen x 1
    constractive = torch.sum(torch.mm(dh**2, w_sum), 0)
    loss = mse + constractive.mul_(lamda)
    return loss

layer_2_output = []

size,s1,s2 = np.shape(input_layer_2)
size = size/2

# Function for training the data
def training1(input_layer_2, n_epochs):
    net1.train()
    for i in range(size):
        a = torch.from_numpy(input_layer_2[n_epochs*507 + i, :,:]).squeeze(0)
        optimizer1.zero_grad()
        out, temp, ww1, ww2, bb1, bb2 = net1(a)
        layer_2_output.append(temp.detach().numpy())
        loss = constractive_loss(out, a, temp, ww1)*5
        loss.backward()
        optimizer1.step()
        print ("[Epoch: %d] [Iter: %d/%d] [Loss: %f]" % (n_epochs +1, i+1, (507), loss.cpu().data.numpy()))

def validating1(data_loader, n_epochs):
    net.eval()
    running_loss = 0
    for i in range(size):
        a = torch.from_numpy(input_layer_2[n_epochs*507 + i, :,:]).squeeze(0)
        out, temp, ww1, ww2, bb1, bb2 = net1(a)
        loss = mmse_loss(out, a)*5
        running_loss += loss.item()
        if n_epochs==1 and not w2:
            w2.append(ww1.detach().numpy())
            w7.append(ww2.detach().numpy())
            b2.append(bb1.detach().numpy())
            b7.append(bb2.detach().numpy())
    return running_loss/(i+1)


# For traning, it is true. For testing, make it 'False' 
isTrain = True
if isTrain:
    epoch = 2
    #arr = []
    for ep in range(epoch):
        training1(input_layer_2, ep)
        gl = validating1(input_layer_2, ep)
        print("loss: " + str(gl))

input_layer_3 = np.array(layer_2_output)
print("2nd Layer Optimization finished!")
##############################################################################

class dnn2(nn.Module):
    
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)
        
    def __init__(self, G_in, w1):
        super(dnn2, self).__init__()    
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, G_in)
    
    # Deep neural network [you are passing data layer-to-layer]
    def forward(self, x):    
        y = F.relu(self.fc1(x))
        x = F.relu(self.fc2(y))
        ww1 = self.fc1.weight
        ww2 = self.fc2.weight
        bb1 = self.fc1.bias
        bb2 = self.fc2.bias
        return x, y, ww1, ww2, bb1, bb2


# initialize the Deep neural network
net2 = dnn2(40, 15)#.cuda()

# Loss function
# Contractive Autoencoder Loss
def constractive_loss1(y_pred, y_true, h, W):
    lamda = 1e-4
    mse = mmse_loss(y_pred, y_true)
    dh = h*(1-h)
    w_sum = torch.sum(Variable(W)**2, dim=1)
    w_sum = w_sum.unsqueeze(1)  # Shape: N_hissen x 1
    constractive = torch.sum(torch.mm(dh**2, w_sum), 0)
    loss = mse + constractive.mul_(lamda)
    return loss

# Optimizer [Adam]
optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.01)

layer_3_output = []

# Function for training the data
def training2(input_layer_4, n_epochs):
    net2.train()
    for i in range(size):
        a = torch.from_numpy(input_layer_3[n_epochs*507 + i, :,:]).squeeze(0)
        optimizer2.zero_grad()
        out, temp, ww1, ww2, bb1, bb2 = net2(a)
        layer_3_output.append(temp.detach().numpy())
        loss = constractive_loss1(out, a, temp, ww1)*5
        loss.backward()
        optimizer2.step()
        print ("[Epoch: %d] [Iter: %d/%d] [Loss: %f]" % (n_epochs +1, i+1, (507), loss.cpu().data.numpy()))

def validating2(data_loader, n_epochs):
    net.eval()
    running_loss = 0
    for i in range(size):
        a = torch.from_numpy(input_layer_3[n_epochs*507 + 1, :,:]).squeeze(0)
        out, temp, ww1, ww2, bb1, bb2 = net2(a)
        loss = mmse_loss(out, a)*5
        running_loss += loss.item()
        if n_epochs==1 and not w3:
            w3.append(ww1.detach().numpy())
            w6.append(ww2.detach().numpy())
            b3.append(bb1.detach().numpy())
            b6.append(bb2.detach().numpy())
    return running_loss/(i+1)

# For traning, it is true. For testing, make it 'False' 
isTrain = True
if isTrain:
    epoch = 2
    #arr = []
    for ep in range(epoch):
        training2(input_layer_3, ep)
        gl = validating2(input_layer_2, ep)
        print("loss: " + str(gl))

input_layer_4 = np.array(layer_3_output)
print("3rd Layer Optimization finished!")

##############################################################################

# ANN layer training

class ann(nn.Module):
    # Weight Initialization [we initialize weights here]
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.bias)

    def __init__(self, G_in, w1):
        super(ann, self).__init__()    
        self.fc1= nn.Linear(G_in, w1)
        self.fc2= nn.Linear(w1, G_in)

    # Deep neural network [you are passing data layer-to-layer]
    def forward(self, x):
        y = F.relu(self.fc1(x))
        x = F.relu(self.fc2(y))
        ww1 = self.fc1.weight
        ww2 = self.fc2.weight
        bb1 = self.fc1.bias
        bb2 = self.fc2.bias
        return x, y, ww1, ww2, bb1, bb2

# initialize the Deep neural network
net3 = ann(15, 75)#.cuda()

# Loss function
loss_mmse = nn.MSELoss()

# Optimizer [Adam]
optimizer3 = torch.optim.Adam(net3.parameters(), lr=0.01)

# Function for training the data
def training3(input_layer_4, n_epochs):
    net3.train()
    for i in range(size):
        a = torch.from_numpy(input_layer_4[n_epochs*507 + i, :,:]).squeeze(0)
        optimizer3.zero_grad()
        out, temp, ww1, ww2, bb1, bb2 = net3(a)
        loss = loss_mmse(out, a)*5
        loss.backward()
        optimizer3.step()
        print ("[Epoch: %d] [Iter: %d/%d] [Loss: %f]" % (n_epochs +1, i+1, (507), loss.cpu().data.numpy()))
    
def validating3(data_loader, n_epochs):
    net.eval()
    running_loss = 0
    for i in range(size):
        a = torch.from_numpy(input_layer_4[n_epochs*507 + 1, :,:]).squeeze(0)
        out, temp, ww1, ww2, bb1, bb2 = net3(a)
        loss = mmse_loss(out, a)*5
        running_loss += loss.item()
        if n_epochs==1 and not w4:
            w4.append(ww1.detach().numpy())
            w5.append(ww2.detach().numpy())
            b4.append(bb1.detach().numpy())
            b5.append(bb2.detach().numpy())
    return running_loss/(i+1)

# For traning, it is true. For testing, make it 'False' 
isTrain = True
if isTrain:
    epoch = 2
    #arr = []
    for ep in range(epoch):
        training3(input_layer_4, ep)
        gl = validating3(input_layer_2, ep)
        print("loss: " + str(gl))

print("ANN Layer Optimization finished!")
print("Pre-training Complete")

#######################################
# Convert List numpy array
weight1 = torch.tensor(np.array(w1)).squeeze(0).numpy()
weight2 = torch.tensor(np.array(w2)).squeeze(0).numpy()
weight3 = torch.tensor(np.array(w3)).squeeze(0).numpy()
weight4 = torch.tensor(np.array(w4)).squeeze(0).numpy()
weight5 = torch.tensor(np.array(w5)).squeeze(0).numpy()
weight6 = torch.tensor(np.array(w6)).squeeze(0).numpy()
weight7 = torch.tensor(np.array(w7)).squeeze(0).numpy()
weight8 = torch.tensor(np.array(w8)).squeeze(0).numpy()

bias1 = torch.tensor(np.array(b1)).squeeze(0).numpy()
bias2 = torch.tensor(np.array(b2)).squeeze(0).numpy()
bias3 = torch.tensor(np.array(b3)).squeeze(0).numpy()
bias4 = torch.tensor(np.array(b4)).squeeze(0).numpy()
bias5 = torch.tensor(np.array(b5)).squeeze(0).numpy()
bias6 = torch.tensor(np.array(b6)).squeeze(0).numpy()
bias7 = torch.tensor(np.array(b7)).squeeze(0).numpy()
bias8 = torch.tensor(np.array(b8)).squeeze(0).numpy()

################################################

import time
# Deep Autoencoder + ANN
start = time.time()

class DNN(nn.Module):
    def __init__(self, ip, h1, h2, h3, h4):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(ip, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, h3)
        self.layer4 = nn.Linear(h3, h4)
        self.layer5 = nn.Linear(h4, h3)
        self.layer6 = nn.Linear(h3, h2)
        self.layer7 = nn.Linear(h2, h1)
        self.layer8 = nn.Linear(h1, ip)
        #self.drop1 = nn.Dropout(0.3)
        self.layer1.weight.data = torch.from_numpy(weight1)
        self.layer2.weight.data = torch.from_numpy(weight2)
        self.layer3.weight.data = torch.from_numpy(weight3)
        self.layer4.weight.data = torch.from_numpy(weight4)
        self.layer5.weight.data = torch.from_numpy(weight5)
        self.layer6.weight.data = torch.from_numpy(weight6)
        self.layer7.weight.data = torch.from_numpy(weight7)
        self.layer8.weight.data = torch.from_numpy(weight8)
        self.layer1.bias.data = torch.from_numpy(bias1)
        self.layer2.bias.data = torch.from_numpy(bias2)
        self.layer3.bias.data = torch.from_numpy(bias3)
        self.layer4.bias.data = torch.from_numpy(bias4)
        self.layer5.bias.data = torch.from_numpy(bias5)
        self.layer6.bias.data = torch.from_numpy(bias6)
        self.layer7.bias.data = torch.from_numpy(bias7)
        self.layer8.bias.data = torch.from_numpy(bias8)

    def forward(self, x):
        x = F.relu(self.layer2(F.relu(self.layer1(x))))
        x = F.relu(self.layer4(F.relu(self.layer3(x))))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer7(F.relu(self.layer6(x))))
        x = self.layer8(x)
        return x 

model = DNN(40, 100, 40, 15, 75)
criterion = nn.MSELoss()
optimizer5 = optim.Adam(model.parameters(), lr = 0.01)

counter = 0
VALIDATION_RMSE=[]
num_epochs = 40
loadtrainingpath = "/home/harshit/Desktop/DNN_VC/WHSP2SPCH_MCEP/batches/Training_complementary_feats"
loadvalidationpath = "/home/harshit/Desktop/DNN_VC/WHSP2SPCH_MCEP/batches/Validation_complementary_feats"
loadtestingpath = "/home/harshit/Desktop/DNN_VC/WHSP2SPCH_MCEP/batches/Testing_complementary_feats"
# Data preprocessing
mainfolder = "/home/harshit/Desktop/pytorch/VCC/"   # main folder where you have all codes
import os
directory_spectrum = mainfolder+'test_spectrum'   # choose a folder where you want to save your test files
if not os.path.exists(directory_spectrum):
    os.makedirs(directory_spectrum)

directory_model = mainfolder+'model_pathDNN_Pre'   # choose a folder where you want to save your model files
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

num_batches_train = 506
num_batches_val = 125
num_batches_test = 34

for epoch in range(1, num_epochs+1):
    random_batch = np.random.permutation(num_batches_train)
    #running_loss = 0.0
    for batch in random_batch:
        path = loadtrainingpath + "/Batch_"+str(batch)+".mat"
        data1 = sio.loadmat(path)
        ip = data1['Feat']
        op = data1['Clean_cent']
        batch_x, batch_y = ip, op

        batch_x = Variable(torch.from_numpy(batch_x))
        optimizer5.zero_grad()
        outputs = model(batch_x)
        batch_y = Variable(torch.from_numpy(batch_y))
        loss = criterion(outputs, batch_y)
        loss.backward()         # Back Propogation
        optimizer5.step()
        #running_loss += loss.item()
        print("Epoch: "+str(epoch)+"  Batch :"+ str(batch) +"  loss = {:.4f}".format(loss))
        
    RMSE=[]
    #running_loss = 0.0
    for batch in range(0,num_batches_val+1):
        path = loadvalidationpath + "/Test_Batch_"+str(batch)+".mat"
        data1 = sio.loadmat(path)
        ip = data1['Feat']
        op = data1['Clean_cent']
        batch_x, batch_y = ip, op
        batch_x = Variable(torch.from_numpy(batch_x))
        outputs= model(batch_x)
        batch_y = Variable(torch.from_numpy(batch_y))
        loss = criterion(outputs, batch_y)
        #running_loss += loss.item()
        print("Validation " + str(epoch)+"  Batch :"+ str(batch) +"  loss = {:.4f}".format(loss))
        RMSE.append(loss.detach().numpy())    
        
    print("Epoch "+str(epoch)+" finished!")
    VALIDATION_RMSE.append(np.average(RMSE))
    model_path = directory_model + "/model" +str(counter)+".pth"
    #torch.save({'epoch': epoch,
    #            'model_state_dict': model.state_dict(),
    #            'optimizer_state_dict': optimizer.state_dict()
    #            }, model_path)
    torch.save(model, model_path)
    counter = counter + 1


print("Optimization finished!")
scipy.io.savemat(mainfolder+"/" + str('Validation_errorDNN.mat'), mdict={'foo': VALIDATION_RMSE})
plt.figure(1)
plt.xlabel("Epochs")
plt.ylabel("Validation RMSE")
plt.plot(VALIDATION_RMSE)
plt.savefig(mainfolder+"/validationerrorDNN_pre.png")

end = time.time()
print("Total time={}".format(end - start))

model_path = directory_model+"/model"+str(39)+".pth"

model = torch.load(model_path)

for i in range(0,num_batches_test+1):
    data1 = sio.loadmat(loadtestingpath + '/Test_Batch_' + str(i)+".mat")         
    batch_x = data1['Feat']   
    batch_x = Variable(torch.from_numpy(batch_x.astype(np.single)))
    pred_spectrum = model(batch_x)
    pred_spec = pred_spectrum.detach().numpy()

    # store clean and predicted mask in file 
    file = directory_spectrum + '/File_'+ str(i)+ '.mat'
    scipy.io.savemat(file,  mdict={'PRED_SPEC': pred_spec   })
    print("file"+str(i))
print("Testing finished!")

