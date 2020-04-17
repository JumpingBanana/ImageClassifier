#
import argparse
import os.path
from os import path
import sys

# --- Import required libray to predict --- #
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from collections import OrderedDict

# Parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument("image_path", action="store", type=str)
parser.add_argument("chk_pth", action="store", type=str, help = ': Checkpoint file to be load')
parser.add_argument("--top_k", action="store", type=int, default=3, help = ': number of top-likely class to be return')
parser.add_argument("--gpu", action="store_true", default=False, help = ': indicate if want to use GPU for training')

args = parser.parse_args()

image_path = args.image_path
chk_pth = args.chk_pth
top_k = 3

# Get setting from input arguments
top_k = args.top_k

# Define function
### TODO: This is duplicate function
def create_classifier(input_size_, hidden_layers_, output_size_ = 102, drop_p_ = 0.5):
    dict = OrderedDict()
    
    # Input to a hidden layer, the first layer
    dict['fc0'] = nn.Linear(input_size_, hidden_layers_[0])
    dict['relu0'] = nn.RReLU(inplace=True)
    dict['dropout0'] = nn.Dropout(p=drop_p_)
    
    # Add a variable number of more hidden_layers
    layer_inx = 1
    layer_sizes = zip(hidden_layers_[:-1], hidden_layers_[1:])
    for layer_size in layer_sizes:
        dict['fc'+str(layer_inx)] = nn.Linear(layer_size[0], layer_size[1])
        dict['relu'+str(layer_inx)] = nn.RReLU(inplace=True)
        dict['dropout'+str(layer_inx)] = nn.Dropout(p=drop_p_)
        # Next layer index
        layer_inx += 1
        
    dict['fc'+str(layer_inx)] = nn.Linear(layer_size[-1], output_size_)
    dict['output'] = nn.LogSoftmax(dim = 1)
    
    return nn.Sequential(dict)

### TODO: This is duplicate function
def create_network(model_name_, hidden_layers_, output_size_ = 102):
    if model_name_ == 'VGG19':
        # Get pretrained model
        model = models.vgg19(pretrained = True)
        # Freeze parameters do we don't do back-propagation through them
        for param in model.parameters():
            param.requires_grad = False
    
        # VGG19's classifier layer input feature size is 25088
#         model.classifier = create_classifier(25088, 102, [4096, 1000, 256], drop_p = 0.4)
        model.classifier = create_classifier(25088, hidden_layers_, output_size_, drop_p_ = 0.5)
        return model
    
    elif model_name_ == 'RestNet50':
        model = models.resnet50(pretrained = True)
        # Freeze parameters do we don't do back-propagation through them
        for param in model.parameters():
            param.requires_grad = False
        # RestNet50's classifier layer input feature size is 2048
#         model.classifier = create_classifier(2048, 102, [1024, 256], drop_p = 0.4)
        model.fc = create_classifier(2048, hidden_layers_, output_size_, drop_p_ = 0.5)
        return model

    elif model_name_ == 'RestNet101':
        model = models.resnet101(pretrained = True)
        # Freeze parameters do we don't do back-propagation through them
        for param in model.parameters():
            param.requires_grad = False
            
        # RestNet101's classifier layer input feature size is 2048
#         model.classifier = create_classifier(2048, 102, [1024, 256], drop_p = 0.4)
        model.fc = create_classifier(2048, hidden_layers_, output_size_, drop_p_ = 0.5)
        return model
    else:
        print("Model: {} not supported .. Abort".format(model_name))
        return None

### TODO: This is duplicate function
def get_device():
    if args.gpu and torch.cuda.is_available():
        device_ = torch.device("cuda")
        print("Using GPU")
    else:
        device_ = torch.device("cpu")
        print("Using CPU")

    return device_

def process_image(image, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image = TF.resize(image, 256)
    
    upper_pixel = (image.height - 224) // 2
    left_pixel = (image.width - 224) // 2
    image = TF.crop(image, upper_pixel, left_pixel, 224, 224)
    
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean, std)
    
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def load_checkpoint(filepath_):
    # Check if file exists
    if(not path.exists(filepath_)): sys.exit()

    checkpoint = torch.load(filepath_)
    
    model_name = checkpoint['model_name']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']

    model = create_network(model_name, hidden_layers, output_size)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.index_to_cat = checkpoint['index_to_cat']
    model.cat_to_name = checkpoint['cat_to_name']

    return model

def predict(image_path_, model_, device_, topk_ =3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path_)
    image = process_image(image)
    
    # Predict in eval mode and no gradient
    with torch.no_grad():
        model_.eval()
        
        # make sure image dimention is right
        image = image.view(1,3,224,224)
        # move image to device
        image = image.to(device_)
        
        # Get model output from forward pass
        log_ps = model.forward(image)
        # Get probability
        ps = torch.exp(log_ps)
        # Get top probability
        top_ps, top_class = ps.topk(topk_, dim = 1)
      
    # Move resulting prediction to 'cpu'
    top_ps, top_class = top_ps.to('cpu'), top_class.to('cpu')
    
    # Convert tensors to numpy
    probs = top_ps.data.numpy().squeeze()
    classes = top_class.data.numpy().squeeze()
    print('Top-{} Probabilitise: {}'.format(topk_, probs))

    # Convert from predicted 'index' to 'name'
    classes = [model.index_to_cat[ii] for ii in classes]
    names = [model.cat_to_name[str(ii)] for ii in classes]
    print("Top-{} Predicted names: {}".format(topk_, names))
  
    return probs, names

if __name__ == "__main__":
    
    # Get device    
    device = get_device()

    # Load Checkpoint and re-create network
    model = load_checkpoint(chk_pth)

    # Move model to device
    model.to(device)
    #print(model)

    # Process image
    print("Image path: {}".format(image_path))
    image = Image.open(image_path)
    image = process_image(image)
    
    # Make prediction
    probs, names = predict(image_path, model, device, top_k)


















