import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import resnet
from train_utils import train_model
from plots import plot_learning_curves

cudnn.benchmark = True


# Loading and transforming the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
dset_loaders = {'train': trainloader, 'val': testloader}
dset_sizes = {'train': len(trainset.train_labels), 'val': len(testset.test_labels)}


# Defining the model
model = resnet.ResNet18()
if torch.cuda.is_available():
    model.cuda()
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


# Training the network
train_outputs = train_model(model, dset_loaders, dset_sizes, criterion, optimizer, num_epochs=350, verbose=2)
plot_learning_curves(train_outputs[0], train_outputs[1], train_outputs[2], train_outputs[3], description='standard_training')
