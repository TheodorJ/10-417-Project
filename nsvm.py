import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F

# Networks trained with this loss function tend to fail at clustering datapoints
# of the same type together
class SpacialSeparation(torch.nn.Module):
    def __init__(self):
        super(SpacialSeparation,self).__init__()

    def forward(self, outputs, labels):
        #print("")
        outputs = outputs
        #print(outputs)
        #print(labels)
        b_n = outputs.shape[0]
        distances = torch.zeros((b_n, b_n))
        for i in range(b_n):
            for j in range(i + 1, b_n):
                distances[i][j] = torch.sqrt(torch.norm(outputs[i] - outputs[j]))
                distances[i][j] *= -1.0 if labels[i] != labels[j] else 5.0
                #distances[i][j] = distances[i][j] if i < j else 0.0

        #print(distances)
        #print(torch.sum(distances))
        #exit(1)
        return torch.sum(distances)

# Turns out, the SVM o[timization problem simplifies down to hinge loss
class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()

    def forward(self, outputs, labels):
        loss = 0.0
        b_n = outputs.shape[0]
        outputs = torch.softmax(outputs, dim=1)
        for i in range(b_n):
            loss += torch.clamp(1 - (outputs[i][labels[i]]), min=0)
        return loss / b_n

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


in_channels = 3
filter_size1 = 5
filter_size2 = 6
inter_channels = 6
out_channels = 16
pool_size = 2
layer_size1 = 120
layer_size2 = 84

num_epochs = 10
lr = 0.001
momentum = 0.9


print("\nTesting for:")
print("  filter_size1 = %s" % filter_size1)
print("  filter_size2 = %s" % filter_size2)
print("  inter_channels = %s" % inter_channels)
print("  out_channels = %s" % out_channels)
print("  pool_size = %s" % pool_size)
print("  layer_size1 = %s" % layer_size1)
print("  layer_size2 = %s" % layer_size2)
print("  num_epochs = %s" % num_epochs)
print("  lr = %s" % lr)
print("  momentum = %s" % momentum)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, filter_size1, padding=1)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, filter_size2, padding=1)
        self.fc1 = nn.Linear(out_channels * 6 * 6, layer_size1)
        self.fc2 = nn.Linear(layer_size1, layer_size2)
        self.fc3 = nn.Linear(layer_size2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, int(x.numel()/x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
criterion = HingeLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)


for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
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
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        b_n = 4
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

print('Finished Training')
