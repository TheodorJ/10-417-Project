import torch
import torchvision
import torchvision.transforms as transforms
import multiprocessing

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


import torch.nn as nn
import torch.nn.functional as F

in_channels = 3
filter_size1 = 5
filter_size2 = 6
inter_channels = 6
out_channels = 16
pool_size = 2
layer_size1 = 120
layer_size2 = 84

num_epochs = 2
lr = 0.001
momentum = 0.9

filter_size1_tries = [4, 5]
filter_size2_tries = [4]
inter_channels_tries = [5]
out_channels_tries = [12, 16, 32]
pool_size_tries = [2]
layer_size1_tries = [80, 120]
layer_size2_tries = [84]
num_epochs_tries = [2]
lr_tries = [0.001]
momentum_tries = [0.9]



                                    
def train_parameters(filter_size1, filter_size2, inter_channels, out_channels, pool_size, layer_size1, layer_size2, num_epochs, lr, momentum):
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
            self.fc1 = nn.Linear(out_channels * 7 * 7, layer_size1)
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

    criterion = nn.CrossEntropyLoss()
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

    print('Finished Training')


    """PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))"""




workers = []
for filter_size1 in filter_size1_tries:
    for filter_size2 in filter_size2_tries:
        for inter_channels in inter_channels_tries:
            for out_channels in out_channels_tries:
                for pool_size in pool_size_tries:
                    for layer_size1 in layer_size1_tries:
                        for layer_size2 in layer_size2_tries:
                            for num_epochs in num_epochs_tries:
                                for lr in lr_tries:
                                    for momentum in momentum_tries:
                                        workers.append(multiprocessing.Process(target=train_parameters, args=(filter_size1, filter_size2, inter_channels, out_channels, pool_size, layer_size1, layer_size2, num_epochs, lr, momentum,)))

for worker in workers:
    worker.start()

for worker in workers:
    worker.join()

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
