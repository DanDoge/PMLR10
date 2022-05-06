# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from model import ConvNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

traindataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=16,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net = ConvNet()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4) # 5e-4, 16 --> 82%
criterion = nn.CrossEntropyLoss()

epoch = 100  # in the paper
for i in range(epoch):
    total_loss = 0
    for j, batch in enumerate(trainloader):
        inputs, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if j % 200 == 199:  # print every 2000 mini-batches
            print(f'[{i + 1}, {j + 1:5d}] loss: {total_loss / 200:.3f}')
            total_loss = 0.0

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    PATH = './log/convnet_{}.pth'.format(i)
    torch.save(net.state_dict(), PATH)

