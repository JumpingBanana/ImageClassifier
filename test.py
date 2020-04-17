#
import argparse
import os.path
from os import path
import sys

# --- Import required libray to train --- #
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

parser.add_argument("data_dir", action="store", type=str)
parser.add_argument("chk_pth", action="store", type=str, help = ': Checkpoint file to be load')
#parser.add_argument("--top_k", action="store", default=3, help = ': number of top-likely class to be return')
parser.add_argument("--gpu", action="store_true", default=False, help = ': indicate if want to use GPU for training')

args = parser.parse_args()

data_dir = args.data_dir
chk_pth = args.chk_pth
top_k = 3
batch_size = 64

# Get setting from input arguments
#top_k = args.top_k

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

def create_dataloader(data_dir_, batch_size_ = batch_size):
    ''' 
    output: train_loaders, valid_loaders, test_loaders
    '''
    print("# Create DataLoader")
    print("## Data Directory: {}".format(data_dir_))
    print("## Batch size: {}".format(batch_size_))

    if path.exists(data_dir_):
        #print('Data directory exists')
        train_dir_ = data_dir_ + '/train'
        if(not path.exists(train_dir_)): sys.exit()
        #print("Train directory: {}".format(train_dir_))

        valid_dir_ = data_dir_ + '/valid'
        if(not path.exists(valid_dir_)): sys.exit()
        #print("Validation directory: {}".format(valid_dir_))

        test_dir_ = data_dir_ + '/test'
        if(not path.exists(test_dir_)): sys.exit()
        #print("Test directory: {}".format(test_dir_))
    else:
        #print('Data directory not exists: {}'.format(data_dir_))
        sys.exit('Data directory not exists -- Abort!')

    train_transforms_ = transforms.Compose([transforms.Resize(255),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms_ = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms_ = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_data_ = datasets.ImageFolder(train_dir_, transform = train_transforms_)
    valid_data_ = datasets.ImageFolder(valid_dir_, transform = valid_transforms_)
    test_data_ = datasets.ImageFolder(test_dir_, transform = test_transforms_)

    # Set batch size to 64 (Default)
    train_loaders_ = torch.utils.data.DataLoader(train_data_, batch_size_, shuffle = True)
    valid_loaders_ = torch.utils.data.DataLoader(valid_data_, batch_size_, shuffle = False)
    test_loaders_ = torch.utils.data.DataLoader(test_data_, batch_size_, shuffle = False)
    
    return train_loaders_, valid_loaders_, test_loaders_

def test_network(model_, device_, test_loaders_, print_every_=5):
    # Turn-off Dropout for testing in Evaluation mode
    model.eval()

    # Define loss function
    criterion = nn.NLLLoss()

    test_loss = 0
    accuracy = 0
    steps = 0

    # For plotting
    test_losses, test_acc = [], []

    # Test loop
    for images_test, labels_test in test_loaders_:
        steps += 1
        # move images and labels to device
        images_test, labels_test = images_test.to(device_), labels_test.to(device_)

        # Do forward pass
        log_ps_test = model_.forward(images_test)
        # Calculate Loss
        loss_test = criterion(log_ps_test, labels_test)

        # Keep track of test loss
        test_loss += loss_test.item()

        # Calculate validation accuracy
        # Get logit from out network, need to do exponential
        ps_test = torch.exp(log_ps_test)
        # Get top prediction of each image in each batch
        top_ps, top_class = ps_test.topk(1, dim = 1)
        # Define Equality
        equality = (top_class == labels_test.view(*top_class.shape))
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        # Collect data for ploting
        test_losses.append(test_loss/len(test_loaders))
        test_acc.append(accuracy/len(test_loaders))

        # After Validation loop is done, print out stats
        print(f"Steps {steps}/{len(test_loaders)}.. "
              f"Test loss: {test_loss/len(test_loaders):.3f}.. "
              f"Test accuracy: {accuracy/len(test_loaders):.3f}")
        print("-------------------------------------------------------\r\n")

if __name__ == "__main__":
    
    # Create DataLoaders for Training, Validation and Test
    train_loaders, valid_loaders, test_loaders = create_dataloader(data_dir)

    # Get device    
    device = get_device()

    # Load Checkpoint and re-create network
    model = load_checkpoint(chk_pth)

    # Move model to device
    model.to(device)
    print(model)

    # Test the network
    test_network(model, device, test_loaders)














