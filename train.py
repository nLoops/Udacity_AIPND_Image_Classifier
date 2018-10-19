import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable
import argparse
from utils import load_model, setup_network, get_current_device, check_accuracy_on_test


def main():
    
    # 1 - get arguments.
    in_args = setup_train_args()
    
    # 2 - setup and apply transforms on datasets.
    train_loader, test_loader, valid_loader, train_data = setup_data(in_args.data_dir)
    
    # 3 - setup network arch and return required objects to function.
    model, criterion, optimizer = setup_network(in_args.arch,in_args.learn_rate,in_args.hidden_layers)
    
    # 4 - start model training and validation.
    train_model(model, criterion, optimizer, train_loader, valid_loader, in_args.epochs, in_args.enable_gpu)
    
    # 5 - save model after training
    save_model(in_args.save_dir,train_data, model, in_args.arch, in_args.learn_rate, in_args.hidden_layers)
    
    # 6 - test model loading function
    ret_model = load_model(in_args.save_dir)
    
    # 7 - test model accuracy on test dataset.
    check_accuracy_on_test(test_loader, ret_model)
    

def setup_train_args():
    """
    Creates Arguments to take input of the user to shape the network
    Returns:
    parse_args
    """
    parseArgs = argparse.ArgumentParser(description = "model train Arguments")
    # start to define args
    parseArgs.add_argument('--data_dir', default = 'flowers', type = str, help = "Your data directory")
    parseArgs.add_argument('--save_dir', default = 'checkpoint.pth', type = str, help = "Model Saved location after training")
    parseArgs.add_argument('--arch', default = 'vgg16', type = str, help = "Model Arch available vgg16, densenet121, alexnet")
    parseArgs.add_argument('--learn_rate', default = 0.0001, type = float, help = "Model learning rate during training")
    parseArgs.add_argument('--hidden_layers', default = 1000, type = int, help = "The Hiddien layers of network")
    parseArgs.add_argument('--epochs', default = 2, type = int , help = "How many loops model will act in training")
    parseArgs.add_argument('--enable_gpu', default = False, action = 'store_true', help = "Enable gpu to accelrate model training")
    
    return parseArgs.parse_args()

def setup_data(data_dir='flowers'):
    """
    Create train, valid, test datasets to allow our model to generalize
    Parameters:
    data directory 
    Returns:
    composed train, test, valid data set to tensors ready for training stage
    """
    
    # 1: define the directories of each set
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # 2: transforms for each set to transfer data into tensors that allow Pytorch to work on data
    # first we declare train_transforms with Augmentation technique, by adding Random Rotation, Resized
    # it will help the network to generalize 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    # in the test_transforms, valid_transforms we don't need any Random values.
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    # 3: Define train, valid and test dataset with their transformations.
    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)

    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    validation_data = datasets.ImageFolder(valid_dir, transform = test_transforms)

    # 4: Define DataLoaders.
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)

    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)

    validloader = torch.utils.data.DataLoader(validation_data, batch_size = 32, shuffle = True)
    
    return trainloader, testloader, validloader, train_data

def train_model(model, criterion, optimizer, train_data_loader, valid_data_loader, epochs = 2, gpu = False):
    """
    Collect all variables and train our model also validate the accuracy on valid_data
    Parameters:
    Model : The Model with new classifier we defined
    Criterion : to measure loss and accuracy
    Optimizer : optimize process
    Train_loader : data of train dirctory prepared for traning
    Valid_loader : data of valid dirctory prepared for validate the accuracy of model to help network to generalize
    Epochs: how many loops we need during training
    Returns:
    NONE just train our model :)
    """
    epochs = epochs
    print_every = 10
    steps = 0

    # change model to device mode.
    device = get_current_device()
    if gpu and device.type == 'cuda':
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)

    # start our training loop
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_data_loader):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            # Forward and Backward passes.
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                
                valid_loss, accuracy = validation(model, valid_data_loader, criterion, optimizer)
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.3f}".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss),
                      'Valid Accuracy: %d %%' % (100 * accuracy))
            
                running_loss = 0
                # make sure that model return to train after eval.
                model.train()
                           
def validation(model, validloader, criterion, optimizer):    
    # put model in eval mode to validate
    model.eval()
    # declare two variables to measure accuracy and loss
    valid_loss = 0
    accuracy = 0
    
    # loop on validloader data
    for ii, (inputs, labels) in enumerate(validloader):
        # increase performance by add zero_grad
        optimizer.zero_grad()
        
        # Set the tensors to device cpu or gpu
        device = get_current_device()
        inputs, labels = inputs.to(device), labels.to(device)
        model.to(device)
        
        with torch.no_grad():
            # calculate loss
            output = model.forward(inputs)
            valid_loss = criterion(output, labels)
            # calculate probabilities
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
            
    # percentage
    valid_loss = valid_loss / len(validloader)
    accuracy = accuracy / len(validloader)
    return valid_loss, accuracy
             
def save_model(path, train_data, model, arch = 'vgg16', lr = 0.0001, hidden_layers = 1000):
    """
    Save trained model
    Parameter:
    checkpoint:  path of saved location
    train_data: train_data to extract classes from it
    model: our trained model
    arch, lr , hidden_layers : to build setup_network
    Returns: 
    NONE just save the model after training
    """
    model.class_to_idx = train_data.class_to_idx
    
    model_checkpoint = {'class_to_idx': model.class_to_idx,
                        'state_dict' : model.state_dict(),
                        'model_arch' : arch,
                        'model_lr' : lr,
                        'model_hidden_layers' : hidden_layers}

    torch.save(model_checkpoint, path)
    
    print("Your model saved successfully")
    
    
    
if __name__== '__main__':
        main()