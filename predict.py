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
from utils import get_current_device, load_model, read_flowers_names


def main():
    # 1 - get arguments
    in_args = setup_predict_args()
    # 2- load model by passing model_path
    model = load_model(in_args.model_path)
    # 3- predict and print most 5 probabilities
    predict(in_args.image_path, model, in_args.topk, in_args.enable_gpu, in_args.names_path)
    
def setup_predict_args():
    """
    Creates required args to predict input images
    Paramtere:
    None
    Returns:
    PraseArgs
    """
    parseArgs = argparse.ArgumentParser(description = "predict images Arguments")
    # start to define args
    parseArgs.add_argument('--image_path', default = 'flowers/test/10/image_07090.jpg', type = str, help = "Images test data dir")
    parseArgs.add_argument('--model_path', default = 'checkpoint.pth', type = str, help = "Saved model path")
    parseArgs.add_argument('--enable_gpu', default = False, action = 'store_true', help = "Enable gpu to accelrate model predicting")
    parseArgs.add_argument('--topk', default = 5, type = int, help = "The number of top probs for the image")
    parseArgs.add_argument('--names_path', default = 'cat_to_name.json', type = str, help = "The path of json file that holds classes names")
    return parseArgs.parse_args()

def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns converted images in Numpy array
    """
    # open image using PIL
    pil_image = Image.open(image)
   
    # Apply the transforms the same we applied on testdata, validdata and convert it to tensor
    apply_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    img_to_tensor = apply_transformations(pil_image)
    # convert processed img from tensor to numpy
    return img_to_tensor.numpy()

def predict(image_path, model, topk=5, gpu = False, names_path = 'cat_to_name.json'):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
        # change model to device mode.
    device = get_current_device()
    if gpu and device.type == 'cuda':
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)
    # process image
    img_torch = process_image(image_path)
    # convert np array to tensor
    image_converted = Variable(torch.from_numpy(img_torch))
    image_converted = image_converted.unsqueeze(0)
    
    with torch.no_grad():
        image_converted = image_converted.type_as(torch.FloatTensor()).to(device)
        output = model.forward(image_converted)
        output = torch.exp(output.cpu())
        
        probs, classes = output.topk(topk)
        
        probs = probs.data.numpy().tolist()[0]
        classes = classes.data.numpy().tolist()[0]
        # read JSON file to get classes name by there numbers
        cat_to_name = read_flowers_names(names_path)
       
        classes_to_idx ={idx:oid for oid,idx in model.class_to_idx.items()}
        classes_name = [cat_to_name[classes_to_idx[i]] for i in classes]
        
        print('Here is top 5 predictions for the image with percentage')
        for c, p in zip(classes_name, probs):
            print("{}  : {:.2%}".format(c,p))
   


if __name__== '__main__':
        main()