import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


import multiprocessing
import math
import csv
import time


from dimensions import *



transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n_test  = len(testset)
n_train = len(trainset)
c_in    = trainset[0][0].shape[0]
h_in    = trainset[0][0].shape[1]
w_in    = trainset[0][0].shape[2]



FILTER_SIZE_1_RANGE = [3,10]
FILTER_SIZE_2_RANGE = [3,10]
INTER_CHANNEL_RANGE = [5,10]
OUT_CHANNEL_RANGE = [12,20]
POOL_SIZE_RANGE = [2, 4]
HIDDEN_LAYER_RANGE = [30,200]
HIDDEN_LAYER_DEPTH = [1,5]

LEARNING_RATE = 0.001
MOMENTUM_RATE = 0.9
NUM_EPOCHS    = 2

RANDOM_SAMPLES = 30

NUM_CORES = 4

RESULT_FOLDER = "results/"
def params_to_filename(hyp):
    fname = RESULT_FOLDER
    for param in hyp:
        fname += str(param) + "-"

    return fname + ".csv"


def build_permutations():

    hyperparameters = []

    for ID in range(RANDOM_SAMPLES):

            layerDepth = np.random.randint(HIDDEN_LAYER_DEPTH[0], high=HIDDEN_LAYER_DEPTH[1])
            hidden_layer = np.random.randint(HIDDEN_LAYER_RANGE[0], high=HIDDEN_LAYER_RANGE[1], size=layerDepth)
            hidden_layer = hidden_layer.tolist()

            inter_channels = np.random.randint(INTER_CHANNEL_RANGE[0], high=INTER_CHANNEL_RANGE[1])
            out_channels   = np.random.randint(OUT_CHANNEL_RANGE[0], high=OUT_CHANNEL_RANGE[1])

            filter_size1 = np.random.randint(FILTER_SIZE_1_RANGE[0], high=FILTER_SIZE_1_RANGE[1])
            filter_size2 = np.random.randint(FILTER_SIZE_2_RANGE[0], high=FILTER_SIZE_2_RANGE[1])

            pool_size = np.random.randint(POOL_SIZE_RANGE[0], high=POOL_SIZE_RANGE[1])


            hyp = (ID, filter_size1, filter_size2, inter_channels, out_channels, pool_size, hidden_layer, LEARNING_RATE, MOMENTUM_RATE,)
            hyperparameters.append(hyp)

    return hyperparameters


hyperparameters = build_permutations()


def train_parameters_range(worker_id):

    for j in range(math.ceil(len(hyperparameters) / NUM_CORES)):
            indx = j * NUM_CORES + worker_id
            if indx < len(hyperparameters):
                hyp = hyperparameters[indx]
                train_parameters(hyp)


def train_parameters(hyp):

    ID, filter_size1, filter_size2, inter_channels, out_channels, pool_size, hidden_layer, lr, momentum = hyp

    print("\n[ID=%d] Testing for:" % ID)
    print("          filter_size1 = %s" % filter_size1)
    print("          filter_size2 = %s" % filter_size2)
    print("          inter_channels = %s" % inter_channels)
    print("          out_channels = %s" % out_channels)
    print("          pool_size = %s" % pool_size)
    print("          hidden_layers = %s" % hidden_layer)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(c_in, inter_channels, filter_size1, padding=1)
            self.pool = nn.MaxPool2d(pool_size, pool_size)
            self.conv2 = nn.Conv2d(inter_channels, out_channels, filter_size2, padding=1)

            c_mid, h_mid, w_mid = conv_dimensions(c_in, h_in, w_in, inter_channels, 1, 1, filter_size1, filter_size1)
            c_mid, h_mid, w_mid = pool_dimensions(c_mid, h_mid, w_mid, pool_size)
            c_out, h_out, w_out = conv_dimensions(c_mid, h_mid, w_mid, out_channels, 1, 1, filter_size2, filter_size2)
            c_out, h_out, w_out = pool_dimensions(c_out, h_out, w_out, pool_size)

            hl = hidden_layer.copy()
            hl.insert(0, c_out * h_out * w_out)
            hl.append(10)

            decoder = []
            for i in range(len(hl)-1):
                decoder.append( nn.Linear(hl[i], hl[i+1]) )
                decoder.append( nn.ReLU() )
            self.decoder = nn.Sequential( *decoder )


        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, int(x.numel()/x.shape[0]))
            x = self.decoder(x)
            return x


    net = Net()


    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)


    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        total_loss   = 0.0
        correct_train = 0

        # Time code found here: https://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
        starttime = int(round(time.time() * 1000))

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss   += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[ID=%d][%d/%d, %5d/%5d] loss: %.3f' %
                        (ID, epoch + 1, NUM_EPOCHS, i + 1, len(trainloader), running_loss / 2000))
                running_loss = 0.0

        # Calculate test loss/accuracy
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        total = 0
        correct = 0
        total_test_loss = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                loss = criterion(outputs, labels)
                total_test_loss += loss
                outputs = net(images)
                loss, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        total_test_loss = total_test_loss / n_test

        endtime = int(round(time.time() * 1000))

        with open(params_to_filename(hyp), "a") as fd:
            fd.write("%d," % epoch)
            for hyper in hyp:
                fd.write(str(hyper) + ",")

            fd.write("%f," % (running_loss / n_train))
            fd.write("%f," % (correct_train / n_train))
            fd.write("%f," % total_test_loss)
            fd.write("%f," % (correct / n_test))
            fd.write("%d,\n" % (endtime - starttime))




    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net_loss = total_loss / n_train
    net_acc  = correct / n_test

    print('Net Loss: %.3f   | Net Acc: %.3f' % (net_loss, net_acc))




workers = []
for i in range(NUM_CORES):
    workers.append(multiprocessing.Process(target=train_parameters_range, args=(i,)))

for worker in workers:
    worker.start()

for worker in workers:
    worker.join()
