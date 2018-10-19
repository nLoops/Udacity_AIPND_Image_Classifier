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



def read_flowers_names(file_path = 'cat_to_name.json'):
    """
    read JSON file that contains flowers names to help us to display flower name instead of class number
    
    Returns:
    list of flowers names
    """
    with open(file_path) as f:
        cat_to_name = json.load(f)
        return cat_to_name
    
def get_current_device():
    """
    Get the current active device cpu or gpu
    Returns:
    current device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def load_model(path):
    """
    Load saved model
    Parameter:
    Path: saved location
    Returns:
    saved model with features.
    """
    checkpoint = torch.load(path)
    model, criterion, optimizer = setup_network(checkpoint['model_arch'], checkpoint['model_lr'], checkpoint['model_hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    print('Your model loaded successfully')
    return model

def setup_network(arch = 'vgg16', lr = 0.0001, hidden_layers = 1000):
    """
    Create Model Network and prepare it for training
    Parameters:
    arch : the model arch we have three types vgg16, densenet121, alexnet
    lr : learning rate that model will perform during his training
    hidden_layers : the count of hidden layers we want our network to have
    gpu : if true we will enable gpu to accelrate training
    Returns:
    model : our model ready for training
    criterion : to measure the loss and accuracy
    optimizer : to optimize the process
    """
    # 1: create dic of archs input layers
    archs = {"vgg16":25088,
            "densenet121":1024,
            "alexnet":9216}
    
    # 2: get input layers
    inputs_layers = archs[arch]
    
    # 3: Download model if not available
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Please make sure to choose one of available architecture")
        
    # 4: freeze all model params we need to train only the classifier params
    for param in model.parameters():
        param.requires_grad = False
    
    # 5: define our classifier network shape
    classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(inputs_layers, hidden_layers)),
                           ('relu1', nn.ReLU()),
                           ('droupout',nn.Dropout(0.5)),
                           ('fc2', nn.Linear(hidden_layers, 500)),
                           ('relu2', nn.ReLU()),
                           ('fc3', nn.Linear(500,102)),
                           ('output', nn.LogSoftmax(dim = 1))]))
    
    # 6: replace model classifier with ours
    model.classifier = classifier
    
    # 7: Declaring Criterion and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer

def check_accuracy_on_test(testloader, model): 
    """
    Test model accuracy on test dataset which is the model never see it and totally a new inputs
    Parameters:
    testloader : prepared dataset with applied torch transforms
    model : our model to check accuracy on the set
    Returns:
    None just print the percentage of the accuracy.
    """
    correct = 0
    total = 0
    device = get_current_device()
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    