import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from dimensions import conv_dimensions, pool_dimensions


def insert_in_channel(old_filter, num_channels):
    old_filter_outc = old_filter.shape[0]
    old_filter_inc = old_filter.shape[1]
    old_filter_width = old_filter.shape[2]
    old_filter_height = old_filter.shape[3]
    new_filter = torch.zeros((old_filter_outc, num_channels, old_filter_width, old_filter_height))

    return torch.cat((old_filter, new_filter), dim=1)

def insert_out_channel(old_filter, num_channels):
    old_filter_outc = old_filter.shape[0]
    old_filter_inc = old_filter.shape[1]
    old_filter_width = old_filter.shape[2]
    old_filter_height = old_filter.shape[3]
    new_filter = torch.zeros((num_channels, old_filter_inc, old_filter_width, old_filter_height))

    return torch.cat((old_filter, new_filter), dim=0)

def expand_conv_kernel(old_filter, pad):
    old_filter_outc = old_filter.shape[0]
    old_filter_inc = old_filter.shape[1]
    old_filter_width = old_filter.shape[2]
    old_filter_height = old_filter.shape[3]
    new_filter = torch.zeros((old_filter_outc, old_filter_inc, old_filter_width + 2*pad, old_filter_height + 2*pad))

    new_filter[:,:,pad:old_filter_width+pad, pad:old_filter_height+pad] = old_filter

    return new_filter

def add_cols_to_matrix(old_matrix, num_cols):
    return torch.cat((old_matrix, torch.zeros((old_matrix.shape[0], num_cols))), dim=1)

def add_rows_to_matrix(old_matrix, num_rows):
    return torch.cat((old_matrix, torch.zeros((num_rows, old_matrix.shape[0]))), dim=0)

def add_output_to_bias(old_bias, num_outs):
    return torch.cat((old_bias, torch.zeros((num_outs))))


import math
import csv
import time

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, int(input.numel()/input.shape[0]))


# Descriptor is of the form:
#  - [({"Conv2d"|"Linear"|"MaxPool"|"ReLU"|"Flatten"|"Reshape"}, weights, bias),]
def descriptor_to_network(descriptor):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.init_descriptor = descriptor

            layers = []
            for layer in descriptor:
                layer_type = layer[0]
                if layer_type == "Conv2d":

                    out_channels = layer[1].shape[1]
                    in_channels = layer[1].shape[0]
                    width = layer[1].shape[2]
                    height = layer[1].shape[2]

                    # Currently only supports square kernels
                    nn_layer = nn.Conv2d(out_channels, in_channels, width, padding=1)
                    if(torch.norm(layer[2]) != 0):
                        nn_layer.weight.data = layer[1]
                        nn_layer.bias.data = layer[2]
                    layers.append(nn_layer)
                elif layer_type == "Linear":
                    in_size = layer[1].shape[1]
                    out_size = layer[1].shape[0]

                    nn_layer = nn.Linear(in_size, out_size)
                    if(torch.norm(layer[2]) != 0):
                        nn_layer.weight.data = layer[1]
                        nn_layer.bias.data = layer[2]
                    layers.append(nn_layer)
                elif layer_type == "MaxPool":
                    mp_size = layer[1].shape[0]
                    layers.append(nn.MaxPool2d(mp_size, mp_size))
                elif layer_type == "ReLU":
                    layers.append(nn.ReLU())
                elif layer_type == "Flatten":
                    layers.append(Flatten())

            self.layers = layers
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            x = self.network(x)
            return x

        def to_descriptor(self):

            new_descriptor = []
            i = 0
            for layer in self.init_descriptor:
                layer_type = layer[0]

                if layer_type in ["Conv2d", "Linear"]:
                    weights = self.layers[i].weight.data
                    bias = self.layers[i].bias.data


                    new_descriptor.append((layer_type, weights, bias))
                else:
                    new_descriptor.append((layer_type,))

                i += 1

            return new_descriptor


    return Net()



c_out, h_out, w_out = conv_dimensions(3, 32, 32, 5, 1, 1, 4, 4)
net = descriptor_to_network([("Conv2d", torch.zeros((5, 3, 4, 4)), torch.zeros((5,))), \
 ("Flatten",), ("ReLU",), \
 ("Linear", torch.zeros((84, c_out * h_out * w_out)), torch.zeros((84,))), ("ReLU",), \
 ("Linear", torch.zeros((10, 84)), torch.zeros((10,)))])



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

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

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
            print('[%d/%d, %5d/%5d] loss: %.3f' %
                    (epoch + 1, 2, i + 1, len(trainloader), running_loss / 2000))
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

    print(correct / total)
    print(total_test_loss)

    total_test_loss = total_test_loss / n_test

    endtime = int(round(time.time() * 1000))

new_descriptor = net.to_descriptor()

new_net = descriptor_to_network(new_descriptor)


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

print(correct / total)
print(total_test_loss)
