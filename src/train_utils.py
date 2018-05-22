import torch
from torch.autograd import Variable
import time
import math


def train_model(model, dset_loaders, dset_sizes, criterion, optimizer, \
                num_epochs=25, verbose=2):
    """
    Method to train a PyTorch neural network with the given parameters for a
    certain number of epochs. Keeps track of the model yielding the best validation
    accuracy during training and returns that model before potential overfitting
    starts happening.
    Records and returns training and validation losses and accuracies over all
    epochs.

    Args:

        model (torch.nn.Module): The neural network model that should be trained.

        dset_loaders (dict[string, DataLoader]): Dictionary containing the training
            loader and test loader: {'train': trainloader, 'val': testloader}

        dset_sizes (dict[string, int]): Dictionary containing the size of the training
            and testing sets. {'train': train_set_size, 'val': test_set_size}

        criterion (PyTorch criterion): PyTorch criterion (e.g. CrossEntropyLoss)

        optimizer (PyTorch optimizer): PyTorch optimizer (e.g. Adam)

        lr_scheduler (PyTorch learning rate scheduler, optional): PyTorch learning rate scheduler

        num_epochs (int): Number of epochs to train for

        verbose (int): Verbosity level. 0 for none, 1 for small and 2 for heavy printouts
    """
    start_time = time.time()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        if verbose > 1:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over mini-batches
            for data in dset_loaders[phase]:
                inputs, labels = data

                # Wrap inputs and labels in Variable
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Zero the parameter gradients
                optimizer.zero_grad()

                preds = model(inputs)

                loss = criterion(preds, labels)

                # Backward pass
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                preds_classes = preds.data.max(1)[1]
                running_corrects += torch.sum(preds_classes == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if verbose > 1:
                print('{} loss: {:.4f}, acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

        if verbose > 1:
            print()

    time_elapsed = time.time() - start_time
    if verbose > 0:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    return train_losses, val_losses, train_accs, val_accs


def lr_finder(model, trainloader, criterion, optimizer, lr_start=1e-8, lr_end=10):
    m = (lr_end / lr_start) ** (1 / (len(trainloader)-1))

    lr = lr_start
    optimizer.param_groups[0]['lr'] = lr

    losses = []
    lrs = []
    for batch_idx, data in enumerate(trainloader):
        inputs, labels = data
        # Wrap inputs and labels in Variable
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels)
        if loss.data[0] > 1000:
            break

        losses.append(loss.data[0])
        lrs.append(lr)

        print('Batch #{}, LR {:.8f} -> loss={:.6f}'.format(batch_idx, lr, loss.data[0]))

        loss.backward()
        optimizer.step()

        lr = lr * m
        optimizer.param_groups[0]['lr'] = lr

    return lrs, losses


def train_phase(model, dset_loaders, dset_sizes, criterion, optimizer, epochs=10,\
                lr_start=1e-8, lr_end=10, mom_start=0.95, mom_end=0.85, verbose=0):
    batches_per_train_epoch = len(dset_loaders['train'])
    total_batches = batches_per_train_epoch * epochs
    step_lr = (lr_end - lr_start) / total_batches
    step_mom = (mom_end - mom_start) / total_batches
    lr = lr_start
    optimizer.param_groups[0]['lr'] = lr
    mom = mom_start
    optimizer.param_groups[0]['momentum'] = mom

    start_time = time.time()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    lrs = []
    moms = []

    for epoch in range(epochs):
        if verbose > 1:
            print('Epoch {}/{}'.format(epoch+1, epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over mini-batches
            for data in dset_loaders[phase]:
                inputs, labels = data

                # Wrap inputs and labels in Variable
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Zero the parameter gradients
                optimizer.zero_grad()

                preds = model(inputs)

                loss = criterion(preds, labels)

                # Backward pass
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    lrs.append(lr)
                    lr += step_lr
                    optimizer.param_groups[0]['lr'] = lr
                    moms.append(mom)
                    mom += step_mom
                    optimizer.param_groups[0]['momentum'] = mom

                running_loss += loss.data[0]
                preds_classes = preds.data.max(1)[1]
                running_corrects += torch.sum(preds_classes == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if verbose > 1:
                print('{} loss: {:.4f}, acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

        if verbose > 1:
            print()

    time_elapsed = time.time() - start_time
    if verbose > 0:
        print('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
    return train_losses, val_losses, train_accs, val_accs, lrs, moms


def one_cycle(model, dset_loaders, dset_sizes, criterion, optimizer, epochs=20, \
              lr_min=1e-3, lr_max=1e-2, mom_min=0.85, mom_max=0.95, annealing_pct=0.1):
    annealing_epochs = int(epochs * annealing_pct)
    up_down_epochs = (epochs - annealing_epochs) // 2

    curves = []

    # Upcycle
    print('##### Computing upcycle phase #####\n')
    curves_upcycle = train_phase(model, dset_loaders, dset_sizes, criterion, optimizer, \
                                 epochs=up_down_epochs, lr_start=lr_min, lr_end=lr_max, \
                                 mom_start=mom_max, mom_end=mom_min, verbose=2)
    curves.append(curves_upcycle)

    # Downcycle
    print('##### Computing downcycle phase #####\n')
    curves_downcycle = train_phase(model, dset_loaders, dset_sizes, criterion, optimizer, \
                                   epochs=up_down_epochs, lr_start=lr_max, lr_end=lr_min, \
                                   mom_start=mom_min, mom_end=mom_max, verbose=2)
    curves.append(curves_downcycle)

    # Annealing
    print('##### Computing annealing phase #####\n')
    curves_annealing = train_phase(model, dset_loaders, dset_sizes, criterion, optimizer, \
                                   epochs=annealing_epochs, lr_start=lr_min, lr_end=lr_min/100, \
                                   mom_start=mom_max, mom_end=mom_max, verbose=2)
    curves.append(curves_annealing)

    print('All three phases done. Concatenating data.')
    concat_curves = []
    for curve_type in range(len(curves_upcycle)):
        c = []
        for phase in range(3):
            c += curves[phase][curve_type]
        concat_curves.append(c)

    return concat_curves
