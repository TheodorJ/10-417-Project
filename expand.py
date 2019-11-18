import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from dimensions import conv_dimensions, pool_dimensions

import math
import csv
import time


def insert_in_channel(old_filter, num_channels, zeros=True):
    old_filter_outc = old_filter.shape[0]
    old_filter_inc = old_filter.shape[1]
    old_filter_width = old_filter.shape[2]
    old_filter_height = old_filter.shape[3]
    new_filter = torch.zeros((old_filter_outc, num_channels, old_filter_width, old_filter_height))
    if(not zeros):
        new_filter.fill_(0.001)

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

def add_cols_to_matrix(old_matrix, num_cols, zeros=True):
    if(zeros):
        return torch.cat((old_matrix, torch.zeros((old_matrix.shape[0], num_cols))), dim=1)
    else:
        return torch.cat((old_matrix, torch.zeros((old_matrix.shape[0], num_cols)).fill_(0.001)), dim=1)

def add_rows_to_matrix(old_matrix, num_rows, zeros=True):
    if(zeros):
        return torch.cat((old_matrix, torch.zeros((num_rows, old_matrix.shape[1]))), dim=0)
    else:
        return torch.cat((old_matrix, torch.zeros((num_rows, old_matrix.shape[1])).fill_(0.001)), dim=0)

def add_output_to_bias(old_bias, num_outs):
    return torch.cat((old_bias, torch.zeros((num_outs))))



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, int(input.numel()/input.shape[0]))

# A new layer always goes at the very end of the model
def insert_layer(old_descriptor):

    old_layer_shape = old_descriptor[-1][1].shape[1]

    new_layer = [("Linear", torch.eye(old_layer_shape), torch.zeros(old_layer_shape)), ("ReLU",)]

    return old_descriptor[:len(old_descriptor)-1] + new_layer + [old_descriptor[-1]]

# A new conv layer should always be appended to the end of the encoder
def insert_conv_layer(old_descriptor):
    # First things first, find the very last convolutional layer
    unchanged = []
    curr_item = old_descriptor[0]
    l = 0
    while(curr_item[0] != "Flatten"):
        unchanged.append(curr_item)

        l += 1
        curr_item = old_descriptor[l]

    # Not -1 because the very last layer is an activation layer

    old_outc = unchanged[-2][1].shape[0]
    old_tensor = torch.zeros((old_outc, old_outc, 5, 5))
    for i in range(old_outc):
        old_tensor[i][i][2][2] = 1.0

    unchanged.append(("Conv2d", old_tensor, torch.zeros(old_outc)))

    return unchanged + [("ReLU",)] + old_descriptor[l:]

def expand_conv_layer(old_descriptor, idx):
    # Find the idx'th convolutional layer
    # First things first, find the very last convolutional layer
    unchanged = []
    curr_item = old_descriptor[0]
    l = 0
    i = 0
    while(curr_item[0] != "Flatten"):


        if curr_item[0] == "Conv2d":
            if i == idx:

                old_filter = curr_item[1]
                old_bias = curr_item[2]
                unchanged.append(("Conv2d", expand_conv_kernel(old_filter, 2), old_bias))
            else:
                unchanged.append(curr_item)
            i += 1
        else:
            unchanged.append(curr_item)

        l += 1
        curr_item = old_descriptor[l]

    return unchanged + old_descriptor[l:]

def insert_hidden_units(old_descriptor, idx):
    # Find the idx'th convolutional layer
    # First things first, find the very last convolutional layer
    unchanged = []
    curr_item = old_descriptor[0]
    l = 0
    i = 0
    while(l < len(old_descriptor)):

        if curr_item[0] == "Linear":
            if i == idx:
                old_filter = curr_item[1]
                old_bias = curr_item[2]
                unchanged.append(("Linear", add_rows_to_matrix(old_filter, old_filter.shape[0]), add_output_to_bias(old_bias, old_bias.shape[0])))
            elif i == idx + 1:
                old_filter = curr_item[1]
                old_bias = curr_item[2]
                unchanged.append(("Linear", add_cols_to_matrix(old_filter, old_filter.shape[1], zeros=False), old_bias))
            else:
                unchanged.append(curr_item)
            i += 1
        else:
            unchanged.append(curr_item)

        l += 1
        if(l < len(old_descriptor)):
            curr_item = old_descriptor[l]

    return unchanged

# Note: This operation (for now) can't be performed on the very last conv layer
# because I'm not quite sure how to resize the following linear layer
def insert_hidden_filter(old_descriptor, idx):
    unchanged = []
    curr_item = old_descriptor[0]
    l = 0
    i = 0
    while(l < len(old_descriptor)):

        input_c = 0
        input_w = 0
        input_h = 0

        new_filter_c = 0
        new_filter_w = 0
        new_filter_h = 0

        if curr_item[0] == "Conv2d":
            if i == idx:
                old_filter = curr_item[1]
                old_bias = curr_item[2]
                unchanged.append(("Conv2d", insert_out_channel(old_filter, old_filter.shape[0]), add_output_to_bias(old_bias, old_bias.shape[0])))
            elif i == idx + 1:
                old_filter = curr_item[1]
                old_bias = curr_item[2]
                new_filter = insert_in_channel(old_filter, old_filter.shape[1], zeros=False).shape
                unchanged.append(("Conv2d", insert_in_channel(old_filter, old_filter.shape[1], zeros=False), old_bias))
            else:
                unchanged.append(curr_item)
            i += 1
        elif curr_item[0] == "Linear" and i - 1 == idx:
            # The conv layer we're expanding feeds into a linear layer, so we need to append
            # columns to it.

            old_filter = curr_item[1]
            old_bias = curr_item[2]
            unchanged.append(("Linear", add_cols_to_matrix(old_filter, old_filter.shape[1], zeros=False), old_bias))

            i += 1

        else:
            unchanged.append(curr_item)

        l += 1
        if(l < len(old_descriptor)):
            curr_item = old_descriptor[l]

    return unchanged

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

                    #print(layer[1].shape)
                    out_channels = layer[1].shape[1]
                    in_channels = layer[1].shape[0]
                    width = layer[1].shape[2]
                    height = layer[1].shape[2]

                    # Currently only supports square kernels
                    nn_layer = nn.Conv2d(out_channels, in_channels, width, padding=int(width/2))
                    if(torch.norm(layer[1]) != 0):
                        nn_layer.weight.data = layer[1]
                        nn_layer.bias.data = layer[2]
                    layers.append(nn_layer)
                elif layer_type == "Linear":
                    in_size = layer[1].shape[1]
                    out_size = layer[1].shape[0]

                    nn_layer = nn.Linear(in_size, out_size)
                    if(torch.norm(layer[1]) != 0):
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



c_out, h_out, w_out = conv_dimensions(3, 33, 33, 5, 1, 2, 4, 4)
net = descriptor_to_network([("Conv2d", torch.zeros((3, 3, 4, 4)), torch.zeros((5,))), ("ReLU",), \
 ("Conv2d", torch.zeros((5, 3, 4, 4)), torch.zeros((5,))), \
 ("ReLU",), ("Flatten",),  \
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


for epoch in range(1):  # loop over the dataset multiple times

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
    print(total_test_loss.item())

    total_test_loss = total_test_loss / n_test

    endtime = int(round(time.time() * 1000))

new_descriptor = net.to_descriptor()

new_descriptor = insert_hidden_filter(new_descriptor, 0)

net = descriptor_to_network(new_descriptor)
