import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 28
plt.style.use('ggplot')
plt.rcParams["axes.grid"] = False
c = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['figure.figsize'] = 8, 4


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, description=None):
    """
    Plots both the training and validation losses and accuracies.

    Args:

        train_losses (list[float]): List of training losses

        val_losses (list[float]): List of validation losses

        train_accs (list[float]): List of training accuracies

        val_accs (list[float]): List of validation accuracies

        description (string, optional): If given, saves plots under this file name
    """
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, c=c[0], label='Train loss')
    plt.plot(val_losses, c=c[1], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(loc='best')
    if description:
        plt.savefig('../plots/' + description + '_loss.png')
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(train_accs, c=c[0], label='Train acc')
    plt.plot(val_accs, c=c[1], label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    if description:
        plt.savefig('../plots/' + description + '_acc.png')
        plt.close()
    else:
        plt.show()

def plot_lr_finder(learning_rates, losses, description=None):
    plt.figure(figsize=(8,4))
    plt.plot(learning_rates, losses)
    plt.xscale('log')
    plt.xlabel('log learning rate')
    plt.ylabel('Loss')
    if description:
        plt.savefig('../plots/' + description + '_lr_finder.png')
        plt.close()
    else:
        plt.show()

def plot_learning_rates(learning_rates, description=None):
    plt.figure(figsize=(8,4))
    plt.plot(learning_rates)
    plt.xlabel('Batch iterations')
    plt.ylabel('Learning rate')
    if description:
        plt.savefig('../plots/' + description + '_lr_1cycle.png')
        plt.close()
    else:
        plt.show()

def plot_momentums(momentums, description=None):
    plt.figure(figsize=(8,4))
    plt.plot(momentums)
    plt.xlabel('Batch iterations')
    plt.ylabel('Momentum')
    if description:
        plt.savefig('../plots/' + description + '_mom_1cycle.png')
        plt.close()
    else:
        plt.show()
