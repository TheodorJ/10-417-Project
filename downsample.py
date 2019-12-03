import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

#import matplotlib.pyplot as plt
import numpy as np


import multiprocessing
import math
import csv
import time


from dimensions import *

UNIFORM_SELECT = True
DS_RATE = 9
BEST_HYP = []

LEARNING_RATE = 0.001
MOMENTUM_RATE = 0.5
NUM_EPOCHS    = 1

NUM_CORES = 4

program_start = time.time()

for i in range(5,DS_RATE):


    DS_H = 2 ** i
    DS_W = 2 ** i

    RANDOM_SAMPLES = 4 * (i-4)

    print("\n-------------------------------------------------------\n")
    print('Training %d Samples on Subsampled Images [%d, %d]' % (RANDOM_SAMPLES, DS_H, DS_W))

    # initial hyperparameter ranges for first call
    transform = transforms.Compose(
                [torchvision.transforms.Resize([DS_H, DS_W], interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    caltech = torchvision.datasets.ImageFolder(root='./data/caltech-101', transform=transform)
    caltechloader = torch.utils.data.DataLoader(caltech, batch_size=1, shuffle=True, num_workers=NUM_CORES)

    NUM_CLASSES = 101
    NUM_SAMPLES = len(caltechloader.dataset)
    TEST_SIZE  = NUM_SAMPLES // 10
    TRAIN_SIZE = NUM_SAMPLES - TEST_SIZE

    trainset, testset = torch.utils.data.random_split(caltech, [TRAIN_SIZE, TEST_SIZE])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=NUM_CORES)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=NUM_CORES)



    CHANNELS_IN = trainset[0][0].shape[0]
    HEIGHT_IN = trainset[0][0].shape[1]
    WIDTH_IN = trainset[0][0].shape[2]

    # these are treated as lo and hi
    FILTER_SIZE_1_RANGE = [3,10]
    FILTER_SIZE_2_RANGE = [3,10]
    INTER_CHANNEL_RANGE = [5,10]
    OUT_CHANNEL_RANGE   = [12,20]
    POOL_SIZE_RANGE     = [2, 4]
    HIDDEN_LAYER_RANGE  = [30,200]
    HIDDEN_LAYER_DEPTH  = [1,5]


    if not UNIFORM_SELECT:
        # these are treated as mu and sigma
        EPS = 1/(i-4)
        _, FS1, FS2, IC, OC, PS, HL, _, _ = BEST_HYP
        HD = len(HL)
        HR = np.average(HL)

        FILTER_SIZE_1_RANGE = [FS1,  5]
        FILTER_SIZE_2_RANGE = [FS2,  5]
        INTER_CHANNEL_RANGE = [IC,   3]
        OUT_CHANNEL_RANGE   = [OC,   3]
        POOL_SIZE_RANGE     = [PS,   3]
        HIDDEN_LAYER_RANGE  = [HR,   20]
        HIDDEN_LAYER_DEPTH  = [HD,   3]




    RESULT_FOLDER = "results/"
    def params_to_filename(hyp):
        fname = RESULT_FOLDER
        for param in hyp:
            fname += str(param) + "-"

        return fname + ".csv"


    def build_permutations():

        hyperparameters = []

        for ID in range(RANDOM_SAMPLES):

                if UNIFORM_SELECT:
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
                
                else:
                    layerDepth = round(np.random.normal(HIDDEN_LAYER_DEPTH[0], HIDDEN_LAYER_DEPTH[1]))
                    layerDepth = max(layerDepth,1)
                    hidden_layer = np.round(np.random.normal(HIDDEN_LAYER_RANGE[0], HIDDEN_LAYER_RANGE[1], size=layerDepth)).astype(int)
                    hidden_layer = hidden_layer.tolist()

                    inter_channels = round(np.random.normal(INTER_CHANNEL_RANGE[0], INTER_CHANNEL_RANGE[1]))
                    out_channels   = round(np.random.normal(OUT_CHANNEL_RANGE[0], OUT_CHANNEL_RANGE[1]))
                    inter_channels = max(inter_channels,1)
                    out_channels   = max(out_channels,1)

                    filter_size1 = round(np.random.normal(FILTER_SIZE_1_RANGE[0], FILTER_SIZE_1_RANGE[1]))
                    filter_size2 = round(np.random.normal(FILTER_SIZE_2_RANGE[0], FILTER_SIZE_2_RANGE[1]))
                    filter_size1 = max(filter_size1,1)
                    filter_size2 = max(filter_size2,1)

                    pool_size = round(np.random.normal(POOL_SIZE_RANGE[0], POOL_SIZE_RANGE[1]))
                    pool_size = max(pool_size,1)

                    hyp = (ID, filter_size1, filter_size2, inter_channels, out_channels, pool_size, hidden_layer, LEARNING_RATE, MOMENTUM_RATE,)
                    hyperparameters.append(hyp)


        return hyperparameters


    hyperparameters = build_permutations()


    def train_parameters_range(worker_id, send_end, train_loader):

        opt_loss = 9999999999
        opt_hyp = 0
        for j in range(math.ceil(len(hyperparameters) / NUM_CORES)):
                indx = j * NUM_CORES + worker_id
                if indx < len(hyperparameters):
                    hyp = hyperparameters[indx]
                    loss = train_parameters(hyp)

                    if loss < opt_loss:
                        opt_loss = loss
                        opt_hyp = hyp
        

        send_end.send( (opt_loss, opt_hyp) )


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
                self.conv1 = nn.Conv2d(CHANNELS_IN, inter_channels, filter_size1, padding=1)
                self.pool = nn.MaxPool2d(pool_size, pool_size)
                self.conv2 = nn.Conv2d(inter_channels, out_channels, filter_size2, padding=1)

                c_mid, h_mid, w_mid = conv_dimensions(CHANNELS_IN, HEIGHT_IN, WIDTH_IN, inter_channels, 1, 1, filter_size1, filter_size1)
                c_mid, h_mid, w_mid = pool_dimensions(c_mid, h_mid, w_mid, pool_size)
                c_out, h_out, w_out = conv_dimensions(c_mid, h_mid, w_mid, out_channels, 1, 1, filter_size2, filter_size2)
                c_out, h_out, w_out = pool_dimensions(c_out, h_out, w_out, pool_size)

                hl = hidden_layer.copy()
                hl.insert(0, c_out * h_out * w_out)
                hl.append(NUM_CLASSES)

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

        LOSS_MAX = 99999999

        try:
            net = Net()
        except:
            print('[ID=%d] Interrupted Training' % ID)
            return LOSS_MAX


        import torch.optim as optim

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)



        for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

            running_loss = 0.0
            total_loss   = 0.0
            correct_train = 0

            # Time code found here: https://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
            starttime = int(round(time.time() * 1000))

            for i, data in enumerate(trainloader):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                try:
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                except:
                    print('[ID=%d] Interrupted Training' % ID)
                    return LOSS_MAX

                # print statistics
                running_loss += loss.item()
                total_loss   += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == labels).sum().item()

                if i % 200 == 199:    # print every 200 mini-batches
                    print('[ID=%d][%d/%d, %5d/%5d] loss: %.3f' %
                            (ID, epoch + 1, NUM_EPOCHS, i + 1, len(trainloader), running_loss / 200))
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

            total_test_loss = total_test_loss / TEST_SIZE

            endtime = int(round(time.time() * 1000))

            '''
            # WRITE TO CSV (OPTIONAL)
            with open(params_to_filename(hyp), "a") as fd:
                fd.write("%d," % epoch)
                for hyper in hyp:
                    fd.write(str(hyper) + ",")

                fd.write("%f," % (running_loss / TRAIN_SIZE))
                fd.write("%f," % (correct_train / TRAIN_SIZE))
                fd.write("%f," % total_test_loss)
                fd.write("%f," % (correct / TEST_SIZE))
                fd.write("%d,\n" % (endtime - starttime))
            '''    



        print('[ID=%d] Finished Training' % ID)

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

        net_loss = total_loss / TRAIN_SIZE
        net_acc  = correct / TEST_SIZE

        return net_loss
        

    best_loss = 9999
    for hyp in hyperparameters:
        loss = train_parameters(hyp)
        if loss < best_loss:
            best_loss = loss
            BEST_HYP = hyp
            UNIFORM_SELECT = False


    if not UNIFORM_SELECT:
        ID, filter_size1, filter_size2, inter_channels, out_channels, pool_size, hidden_layer, lr, momentum = BEST_HYP

        print("\nBEST RESULT")
        print("\n[ID=%d] Loss: %.3f" % (ID, loss) ) 
        print("          filter_size1 = %s" % filter_size1)
        print("          filter_size2 = %s" % filter_size2)
        print("          inter_channels = %s" % inter_channels)
        print("          out_channels = %s" % out_channels)
        print("          pool_size = %s" % pool_size)
        print("          hidden_layers = %s" % hidden_layer)

    program_end = time.time()

    print('\nNet Time: %.3f' % (program_end - program_start))