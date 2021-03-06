from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from fastai.vision.all import *
import matplotlib.pyplot as plt
from IPython import display



def get_dls(path, size, batch_size, num_workers, device):
    """Dataloaders for dataset from path.
    
    """
    source = untar_data(path)

    resize_ftm = Resize(size)
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=GrandparentSplitter(valid_name='val'),
                       get_items=get_image_files, get_y=parent_label,
                       item_tfms=[resize_ftm])
    return dblock.dataloaders(source, path=source, device=device, bs=batch_size, num_workers=num_workers)

def prepare_train_and_val_dls(path, batch_size, device, size=160, num_workers=0):
    """Create and shuffle training and validation dataloaders.
    
    """
    dloaders = get_dls(path, size=size, batch_size=batch_size, num_workers=num_workers, device=device)
    trainloader = dloaders.train
    trainloader = trainloader.new(shuffle=True)
    valloader = dloaders.valid
    valloader = valloader.new(shuffle=True, bs=batch_size*2)
    return trainloader, valloader

def accuracy_and_loss(dataloader, net, device, criterion):
    """Accuracy and loss for network inference.
    
    """
    current_loss = 0.0
    steps = 0
    total = 0
    correct = 0
    
    for data in dataloader:
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            current_loss += loss.cpu().numpy()
            steps += 1

    current_loss /= steps
    accuracy = correct / total
    return accuracy, current_loss

def show_plot(data_arrays, path):
    """
    Drawing graphs of the learning process.
    
    Includes: train loss, train accuracy, validation loss, validation accuracy.
    """
    y_name=["Loss", "Accuracy"]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    for plot_num, ax in enumerate(axs.flat):
        ax.plot(data_arrays[plot_num], label=f'Train {y_name[plot_num].lower()}')
        ax.plot(data_arrays[plot_num+2], label=f'Validation {y_name[plot_num].lower()}')
        ax.set_title(y_name[plot_num])
        ax.set_xlabel("Epoch number")
        ax.set_ylabel(y_name[plot_num])
        ax.grid(True)
        ax.legend()


    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


    
def train(net, optimizer, criterion, epoch_num, trainloader, valloader, device, graph_path):
    """Neural network training process.
    
    """
    val_loss_history = []
    val_accuracy_history = []
    train_accuracy_history = []
    train_loss_history = []
    
    for epoch in tqdm(range(epoch_num)):  # loop over the dataset multiple times
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Train accuracy
        current_train_accuracy, current_train_loss = accuracy_and_loss(trainloader, net, device, criterion)     
        train_accuracy_history.append(current_train_accuracy)
        train_loss_history.append(current_train_loss)

        # Validation accuracy and loss
        current_val_accuracy, current_val_loss = accuracy_and_loss(valloader, net, device, criterion)     
        val_loss_history.append(current_val_loss)
        val_accuracy_history.append(current_val_accuracy)
        
        # visulization:
        if (epoch+1)%5 == 0:
            show_plot(data_arrays=[train_loss_history, train_accuracy_history, 
                                   val_loss_history, val_accuracy_history], path=graph_path)        
        
        """
        print('Current train loss: %f' % current_train_loss)
        print('Current train accuracy: %f' % current_train_accuracy)

        print('Current validation loss: %f' % current_val_loss)
        print('Current validation accuracy: %f' % current_val_accuracy)
        """
    return net, {"val losses": val_loss_history, 
                 "val accuracy": val_accuracy_history, 
                 "train losses": train_loss_history, 
                 "train accuracy": train_accuracy_history}

def test_metrics(net, device, testloader, out_channels):    
    """
    Metrics on test set.
    
    Includes: accuracy, confusion matrix, precision, recall, f1.    
    """
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    correct = 0
    total = 0
    confusion_matrix_counted = torch.zeros(out_channels, out_channels)

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # true_positive + true_negative

            true_positive += ((predicted == labels) == 1).sum().item()            
            false_positive += ((predicted == 1) != labels).sum().item()
            false_negative += ((predicted == 0) != labels).sum().item()

            confusion_matrix_counted[labels, predicted] += 1
                
    confusion_matrix_counted /= confusion_matrix_counted.sum(axis=1)
    accuracy = correct / total
    
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = 2*precision*recall/(precision+recall)
   
    return accuracy, confusion_matrix_counted, precision, recall, f1
