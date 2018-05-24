import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import numpy as np

import resnet
from train_utils import one_cycle
from plots import plot_learning_curves, plot_learning_rates, plot_momentums

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


# Try different parameters
# standard, more annealing, heavy boosting, moderate boosting, constant momentum, moderate boosting constant momentum, slow boosting, standard with annealing
params = [{'lr_min': 0.001, 'lr_max': 0.01, 'mom_min': 0.85, 'mom_max': 0.95, 'annealing_pct': 0.1, 'epochs': 90},
          {'lr_min': 0.001, 'lr_max': 0.01, 'mom_min': 0.85, 'mom_max': 0.95, 'annealing_pct': 0.5, 'epochs': 90},
          {'lr_min': 0.1, 'lr_max': 3, 'mom_min': 0.85, 'mom_max': 0.95, 'annealing_pct': 0.1, 'epochs': 50},
          {'lr_min': 0.01, 'lr_max': 0.1, 'mom_min': 0.85, 'mom_max': 0.95, 'annealing_pct': 0.1, 'epochs': 70},
          {'lr_min': 0.001, 'lr_max': 0.01, 'mom_min': 0.9, 'mom_max': 0.9, 'annealing_pct': 0.1, 'epochs': 90},
          {'lr_min': 0.01, 'lr_max': 0.1, 'mom_min': 0.9, 'mom_max': 0.9, 'annealing_pct': 0.1, 'epochs': 70},
          {'lr_min': 0.001, 'lr_max': 0.1, 'mom_min': 0.85, 'mom_max': 0.95, 'annealing_pct': 0.1, 'epochs': 90},
          {'lr_min': 0.01, 'lr_max': 0.01, 'mom_min': 0.9, 'mom_max': 0.9, 'annealing_pct': 0.1, 'epochs': 50}]
for p in params:

    # Defining the model
    model = resnet.ResNet18()
    if torch.cuda.is_available():
        model.cuda()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # Training the network
    curves = one_cycle(model, dset_loaders, dset_sizes, criterion, optimizer, epochs=p['epochs'],
                       lr_min=p['lr_min'], lr_max=p['lr_max'], mom_min=p['mom_min'],
                       mom_max=p['mom_max'], annealing_pct=p['annealing_pct'], verbose=2)

    description = 'lrmin{}_lrmax{}_mmin{}_mmax{}_pct{}_e{}'.format(p['lr_min'], p['lr_max'],
                                                                   p['mom_min'], p['mom_max'],
                                                                   p['annealing_pct'], p['epochs'])
    plot_learning_curves(curves[0], curves[1], curves[2], curves[3], description=description)
    plot_learning_rates(curves[4], description=description)
    plot_momentums(curves[5], description=description)

    print(description)
    print('train loss: {}, val loss: {}, train acc: {}, val_acc:{}'.format(curves[0][-1],
                                                                           curves[1][-1],
                                                                           curves[2][-1],
                                                                           curves[3][-1]))

    # Save all curves to disk for later plotting
    for i in range(6):
        np.save('../learning_curves/' + description + '-' + str(i), np.array(curves[i]))
