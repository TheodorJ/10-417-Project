import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from dimensions import conv_dimensions, pool_dimensions

import math
import csv
import time
import torch.multiprocessing as multiprocessing

import torch.optim as optim

import argparse

NUM_CORES = 1
BEAM_WIDTH = 2

birthday = int(round(time.time() * 1000))

MOMENTUM = 0.5

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

    maxp_layer = unchanged[-1]
    unchanged = unchanged[:len(unchanged) - 1]

    print(summarize_descriptor(unchanged))
    old_outc = unchanged[-2][1].shape[0]
    old_tensor = torch.zeros((old_outc, old_outc, 5, 5))
    for i in range(old_outc):
        old_tensor[i][i][2][2] = 1.0

    unchanged.append(("Conv2d", old_tensor, torch.zeros(old_outc)))

    return unchanged + [("ReLU",), maxp_layer] + old_descriptor[l:]

# A new conv layer should always be appended to the end of the encoder
def insert_conv_layer_maxpool(old_descriptor):
    # First things first, find the very last convolutional layer
    unchanged = []
    curr_item = old_descriptor[0]
    l = 0
    while(curr_item[0] != "Flatten"):
        if(curr_item[0] == "Conv2d"):
            unchanged.append(("Conv2d", torch.zeros(curr_item[1].shape), torch.zeros((curr_item[2].shape[0]+1,))))
        else:
            unchanged.append(curr_item)

        l += 1
        curr_item = old_descriptor[l]

    # Not -1 because the very last layer is an activation layer

    maxp_layer = unchanged[-1]
    #unchanged = unchanged[:len(unchanged) - 1]

    old_outc = unchanged[-3][1].shape[0]
    old_tensor = torch.zeros((old_outc, old_outc, 5, 5))
    for i in range(old_outc):
        old_tensor[i][i][2][2] = 1.0

    unchanged.append(("Conv2d", old_tensor, torch.zeros(old_outc)))
    unchanged.append(("ReLU",))
    unchanged.append(("MaxPool", torch.zeros(2)))

    for layer in old_descriptor[l:]:
        l += 1
        if layer[0] == "Linear":
            unchanged.append(("Linear", torch.zeros((layer[1].shape[0], int(layer[1].shape[1]/4))), torch.zeros((layer[2].shape[0]+1,))))

            break
        else:
            unchanged.append(layer)

    return unchanged + old_descriptor[l:]

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

def summarize_descriptor(descriptor):
    s = "\""
    for layer in descriptor:
        s += layer[0] + ":"
        if layer[0] in ["Conv2d", "Linear"]:
            s += "(" + str(layer[1].shape) + ";" + str(layer[2].shape) + ")"

        s += "->"
    s += "\""

    return s

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
def descriptor_to_network(descriptor, ignore_values=False):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.init_descriptor = descriptor

            self.transfer_occured = True

            layers = []
            i = -1
            for layer in descriptor:
                i += 1
                layer_type = layer[0]
                if layer_type == "Conv2d":

                    #print(layer[1].shape)
                    out_channels = layer[1].shape[1]
                    in_channels = layer[1].shape[0]
                    width = layer[1].shape[2]
                    height = layer[1].shape[2]

                    # Currently only supports square kernels
                    nn_layer = nn.Conv2d(out_channels, in_channels, width, padding=int(width/2))
                    if(not ignore_values and layer[1].shape[0] == layer[2].shape[0]):
                        nn_layer.weight.data = layer[1]
                        nn_layer.bias.data = layer[2]
                    else:
                        self.transfer_occured = False
                    layers.append(nn_layer)
                elif layer_type == "Linear":
                    in_size = layer[1].shape[1]
                    out_size = layer[1].shape[0]

                    nn_layer = nn.Linear(in_size, out_size)
                    if(not ignore_values and layer[1].shape[0] == layer[2].shape[0]):
                        nn_layer.weight.data = layer[1]
                        nn_layer.bias.data = layer[2]
                    else:
                        self.transfer_occured = False
                    layers.append(nn_layer)
                elif layer_type == "Dropout":
                    layers.append(nn.Dropout(p=0.3))
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
                elif layer_type == "MaxPool":
                    mp_size = self.layers[i].kernel_size
                    new_descriptor.append((layer_type, torch.zeros(mp_size)))
                else:
                    new_descriptor.append((layer_type,))

                i += 1

            return new_descriptor

    return Net()

# Given a descriptor, return every possible descriptor that can be made by making
# a single change to this descriptor
def generate_all_modifications(descriptor):

    mutations = []

    conv_index = 0
    line_index = 0
    for layer in descriptor[:len(descriptor)-1]:
        layer_type = layer[0]
        if layer_type == "Conv2d":

            mutations += [insert_hidden_filter(descriptor, conv_index)]
            mutations += [expand_conv_layer(descriptor, conv_index)]

            conv_index += 1
        elif layer_type == "Linear":
            mutations += [insert_hidden_units(descriptor, line_index)]

            line_index += 1

    mutations += [insert_conv_layer(descriptor)]
    mutations += [insert_conv_layer_maxpool(descriptor)]
    mutations += [insert_layer(descriptor)]

    return mutations

def train_descriptor(descriptor, trainloader, num_epochs=1, lr=0.01, momentum=0.5):

    net = descriptor_to_network(descriptor, ignore_values=args.no_knowledge_transfer)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    if(not net.transfer_occured):
        num_epochs *= 3

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        total_loss   = 0.0
        correct_train = 0


        #lr /= 10

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
                        (epoch + 1, num_epochs, i + 1, len(trainloader), running_loss / 2000))
                running_loss = 0.0
                break


    total = 0
    correct = 0
    total_test_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss
            outputs = net(images)
            loss, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    total_test_loss = total_test_loss.item() / n_test

    print("Train Accuracy after training:")
    print(correct / total)

    return net.to_descriptor()

def evaluate_descriptor(descriptor, testloader):
    # Calculate test loss/accuracy
    net = descriptor_to_network(descriptor, ignore_values=False)
    total = 0
    correct = 0
    total_test_loss = 0
    criterion = nn.CrossEntropyLoss()
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

    total_test_loss = total_test_loss.item() / n_test

    return correct / total,  total_test_loss

def thread_manage_mutations(mutations, trainloader, testloader, lr, momentum):
    # for each mutation, train it calculate its validation accuracy
    scores = []
    new_mutations = []

    while(mutations != []):
        if(NUM_CORES == 1):
            return_dict = {}
            mut = mutations[0]
            beam_search_thread(mut, trainloader, testloader, 0, return_dict, lr, momentum)
            mutations = mutations[1:]

            for i in return_dict.keys():
                worker_acc, worker_loss, worker_mut = return_dict[i]
                scores.append(worker_acc)
                new_mutations.append(worker_mut)
            pass
        else:
            workers = []
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            for w in range(NUM_CORES): # -1 so that we only detach once
                if mutations == []:
                    break

                print("Evaluating a mutation...")
                mut = mutations[0]
                workers.append(multiprocessing.Process(target=beam_search_thread, args=(mut, trainloader, testloader, w, return_dict,)))
                mutations = mutations[1:]

            for worker in workers:
                worker.start()

            for worker in workers:
                worker.join()

            for i in return_dict.keys():
                worker_acc, worker_loss, worker_mut = return_dict[i]
                scores.append(worker_acc)
                new_mutations.append(worker_mut)


    return scores, new_mutations

def beam_search_thread(mut, trainloader, testloader, i, return_dict, lr=0.01, momentum=0.5):
    print(summarize_descriptor(mut))
    new_mut = train_descriptor(mut, trainloader, num_epochs=1, lr=lr, momentum=momentum)
    val_acc, val_loss = evaluate_descriptor(new_mut, testloader)
    print("val_acc = %f" % val_acc)
    return_dict[i] = (val_acc, val_loss, new_mut)

def beam_search(descriptor, beam_width, trainloader, testloader):
    # For some damn reason the line below causes all subsequent calls to
    # loss.backward() to crash
    #first_acc, _ = evaluate_descriptor(descriptor, testloader)
    descriptor = train_descriptor(descriptor, trainloader, num_epochs=1, lr=0.001, momentum=MOMENTUM)


    #original_acc = 0.0
    original_acc, _ = evaluate_descriptor(descriptor, testloader)
    print("Accuracy to beat: %f" % original_acc)

    print("Generating all modifications...")
    # First, we get all of the mutations of this descriptor
    mutations = generate_all_modifications(descriptor)

    lr = 0.001
    momentum = MOMENTUM + 0.2

    print("Entering manager routine...")
    # Now for each mutation, train it calculate its validation accuracy
    scores, mutations = thread_manage_mutations(mutations, trainloader, testloader, lr=lr, momentum=momentum)

    best_scores, indices = torch.topk(torch.Tensor(scores), beam_width)
    print(best_scores)

    best_mutations = []
    for i in range(len(indices)):
        if best_scores[i] > original_acc or True:
            best_mutations.append((best_scores[i], mutations[indices[i]]))

    now = int(round(time.time() * 1000))
    with open(args.results_file, "a") as fd:
        for bm in best_mutations:
            fd.write("0, %d, %f, %s\n" % ((now - birthday), bm[0], summarize_descriptor(bm[1])))

    round_num = 1
    while(best_mutations != []):
        #momentum += 0.2
        #lr /= 10
        all_mutations = []
        all_scores = []
        for bm_score, bm in best_mutations:
            print("Generating modifications...")
            mutations = generate_all_modifications(bm)

            print("Entering manager")
            # Now for each mutation, train it calculate its validation accuracy
            scores, mutations = thread_manage_mutations(mutations, trainloader, testloader, lr=lr, momentum=momentum)

            # Safe because there should always be greater than beam_width
            # models
            best_scores, indices = torch.topk(torch.Tensor(scores), beam_width)

            for i in range(len(indices)):
                if best_scores[i] > bm_score:
                    all_mutations.append(mutations[indices[i]])
                    all_scores.append(best_scores[i])

        if len(all_scores) < beam_width:
            best_scores = all_scores
            indices = range(len(all_scores))
        else:
            best_scores, indices = torch.topk(torch.Tensor(all_scores), beam_width)

        best_mutations = []
        for i in range(len(indices)):
            print("Round %d: %s" % (round_num, str(best_scores[i])))
            best_mutations.append((best_scores[i], all_mutations[indices[i]]))

        now = int(round(time.time() * 1000))

        with open(args.results_file, "a") as fd:
            for bm in best_mutations:
                fd.write("%d, %d, %f, %s\n" % (round_num, (now - birthday), bm[0], summarize_descriptor(bm[1])))

        round_num += 1

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_knowledge_transfer", action="store_true")
    parser.add_argument("--two_conv", help="2-layer convolutional network with 1 maxpool layer", action="store_true")
    parser.add_argument("--two_conv_mp", help="2-layer convolutional network with 2 maxpool layers", action="store_true")
    parser.add_argument("--results_file", help="filename to output results to", default="beam_search_results.csv")
    args = parser.parse_args()


    with open(args.results_file, "a") as fd:
        fd.write("Round Number, Time Taken (ms), Accuracy, Model summary\n")

    if(args.two_conv_mp):
        c_out, h_out, w_out = conv_dimensions(3, 8, 8, 32, 1, 2, 5, 5)
        desc = [("Conv2d", torch.zeros((7, 3, 5, 5)), torch.zeros((6,))), ("ReLU",),("MaxPool", torch.zeros((2))),\
            ("Conv2d", torch.zeros((32, 7, 5, 5)), torch.zeros((6,))), ("ReLU",), \
          ("MaxPool", torch.zeros((2))), ("Flatten",),  \
         ("Linear", torch.zeros((84, c_out * h_out * w_out)), torch.zeros((85,))), ("ReLU",), \
         ("Linear", torch.zeros((84, 84)), torch.zeros((85,))), ("ReLU",), \
         ("Linear", torch.zeros((10, 84)), torch.zeros((11,)))]
    elif(args.two_conv):
        c_out, h_out, w_out = conv_dimensions(3, 16, 16, 32, 1, 2, 5, 5)
        desc = [("Conv2d", torch.zeros((7, 3, 5, 5)), torch.zeros((6,))), ("ReLU",), \
            ("Conv2d", torch.zeros((32, 7, 5, 5)), torch.zeros((6,))), ("ReLU",), \
          ("MaxPool", torch.zeros((2))), ("Flatten",),  \
         ("Linear", torch.zeros((84, c_out * h_out * w_out)), torch.zeros((85,))), ("ReLU",), \
         ("Linear", torch.zeros((10, 84)), torch.zeros((11,)))]
    else:
        c_out, h_out, w_out = conv_dimensions(3, 16, 16, 7, 1, 2, 5, 5)
        desc = [("Conv2d", torch.zeros((7, 3, 5, 5)), torch.zeros((8,))), ("ReLU",), \
          ("MaxPool", torch.zeros((2))), ("Flatten",),  \
         ("Linear", torch.zeros((84, c_out * h_out * w_out)), torch.zeros((85,))), ("ReLU",), \
         ("Linear", torch.zeros((10, 84)), torch.zeros((11,)))]
        MOMENTUM = 0.5


    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    n_test  = len(testset)
    n_train = len(trainset)
    c_in    = trainset[0][0].shape[0]
    h_in    = trainset[0][0].shape[1]
    w_in    = trainset[0][0].shape[2]

    beam_search(desc, BEAM_WIDTH, trainloader, testloader)
    #trained_descriptor = train_descriptor(desc, trainloader, criterion)

    #print(evaluate_descriptor(trained_descriptor, testloader, criterion))
