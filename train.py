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
parser.add_argument("--save_dir", "-s", type=str, required=False, help = ': directory to save checkpoints')
parser.add_argument("--class_map", "-c", type=str, default='cat_to_name.json', required=False, help = ': JSON file that mpas the class values to category names')
parser.add_argument("--arch", "-a", type=str, required=False, help = ': type of architecture to be train')
parser.add_argument("--batch_size", "-b", type=int, default=64, required=False, help = ': indicate batch size')
parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, required=False, help = ': indicate learning rate')
parser.add_argument("--hidden_units", "-u", nargs="*", type=int, default=[1024], required=False, help = ': indicate number of hidden unit')
parser.add_argument("--epochs", "-e", type=int, default=10, required=False, help =': indicate number of epochs')
parser.add_argument("--gpu", action="store_true", default=False, help = ': indicate if want to use GPU for training')

args = parser.parse_args()

# Default parameter
learn_rate = 0.01
epochs = 10
batch_size = 64
save_dir = 'ImgClassifier.pth'
model_name = 'RestNet101'
output_size = 102

# Get training information from input arguments
data_dir = args.data_dir
batch_size = args.batch_size
hidden_layers = args.hidden_units
learn_rate = args.learning_rate
epochs = args.epochs
save_dir = args.save_dir
model_name = args.arch
json_file = args.class_map

# Category to name relation
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Index to category relation
index_to_cat = [1 ,10 , 100, 101, 102, 11, 12, 13, 14, 15, 16, 17, 18, 19,
               2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
               3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
               4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
               5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
               6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
               7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
               8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
               9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

with open(json_file, 'r') as f:
    cat_to_name = json.load(f)

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

def create_network(model_name_, hidden_layers_, output_size_ = 102):
    if model_name_ == 'VGG19':
        # Get pretrained model
        print("Pretrain model: {}".format(model_name_))
        model = models.vgg19(pretrained = True)
        # Freeze parameters do we don't do back-propagation through them
        for param in model.parameters():
            param.requires_grad = False
    
        # VGG19's classifier layer input feature size is 25088
#         model.classifier = create_classifier(25088, 102, [4096, 1000, 256], drop_p = 0.4)
        model.classifier = create_classifier(25088, hidden_layers_, output_size_, drop_p_ = 0.5)
        return model
    
    elif model_name_ == 'RestNet50':
        print("Pretrain model: {}".format(model_name_))
        model = models.resnet50(pretrained = True)
        # Freeze parameters do we don't do back-propagation through them
        for param in model.parameters():
            param.requires_grad = False
        # RestNet50's classifier layer input feature size is 2048
#         model.classifier = create_classifier(2048, 102, [1024, 256], drop_p = 0.4)
        model.fc = create_classifier(2048, hidden_layers_, output_size_, drop_p_ = 0.5)
        return model

    elif model_name_ == 'RestNet101':
        print("Pretrain model: {}".format(model_name_))
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

def get_device():
    if args.gpu and torch.cuda.is_available():
        device_ = torch.device("cuda")
        print("Using GPU")
    else:
        device_ = torch.device("cpu")
        print("Using CPU")

    return device_

def train_network(model_, epochs_, learn_rate_, train_loaders_,
                valid_loaders_, device_, print_every_ = 5):
    print("## Start training ...")
    # Define loss function
    criterion = nn.NLLLoss()

    # Define Optimizer
    if model_name == 'VGG19':
        optimizer = optim.Adam(model_.classifier.parameters(), lr = learn_rate_)
    else:
        optimizer = optim.Adam(model_.fc.parameters(), lr = learn_rate_)

    # Set Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Parameter for training
    steps = 0
    training_loss = 0

    # For plotting
    train_losses, valid_losses, acc_trace = [], [], []

    # Loop through epochs
    for epoch in range(epochs_):
        steps = 0
        # Decay Learning Rate
        scheduler.step()
    
        for images, labels in train_loaders_:
            # increase steps to keep track of number of bacth
            steps += 1
            # move images and labels to device
            images, labels = images.to(device_), labels.to(device_)
            # Reset Optimizer gradient
            optimizer.zero_grad()
        
            # Forward pass
            log_ps = model_.forward(images)
            # Calculate loss from criterion
            loss = criterion(log_ps, labels)
            # Calculate weight gradient from back-propagation process
            loss.backward()
            # Update weight, using optimizer
            optimizer.step()
        
            # Keep track of training loss
            training_loss += loss.item()
        
            # if steps is multiple of print_every then do validation test and print stats
            if steps % print_every_ == 0:
                # Turn-off Dropout
                model_.eval()
            
                valid_loss = 0
                accuracy = 0
            
                # Validation loop
                for images_valid, labels_valid in valid_loaders_:
                    # move images and labels to device
                    images_valid, labels_valid = images_valid.to(device_), labels_valid.to(device_)
                
                    # Do forward pass
                    log_ps_valid = model_.forward(images_valid)
                    # Calculate Loss
                    loss_valid = criterion(log_ps_valid, labels_valid)
                
                    # Keep track of validation loss
                    valid_loss += loss_valid.item()
                
                    # Calculate validation accuracy
                    # Get logit from out network, need to do exponential
                    ps_valid = torch.exp(log_ps_valid)
                    # Get top prediction of each image in each batch
                    top_ps, top_class = ps_valid.topk(1, dim = 1)
                    # Define Equality
                    equality = (top_class == labels_valid.view(*top_class.shape))
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            
                # Collect data for ploting
                train_losses.append(training_loss/print_every_)
                valid_losses.append(valid_loss/len(valid_loaders_))
                acc_trace.append(accuracy/len(valid_loaders_))
            
                # After Validation loop is done, print out stats
                print("Learning Rate: {}".format(scheduler.get_lr()))
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Steps {steps}/{len(train_loaders_)}.. "
                      f"Training loss: {training_loss/print_every_:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loaders_):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loaders_):.3f}")
                print("-------------------------------------------------------\r\n")
            
                # Reset Training loss
                training_loss = 0
                # Set model back to Training mode
                model_.train()

def save_checkpoint(model_name_, output_size_, hidden_layers_,
                    model_, index_to_cat_, cat_to_name_, filepath_):
    # Move model to CPU
    model_.to('cpu')
    checkpoint = {  'model_name': model_name_,
                    'output_size': output_size_,
                    'hidden_layers': hidden_layers_,
                    'state_dict': model_.state_dict(),
                    'index_to_cat': index_to_cat_,
                    'cat_to_name':  cat_to_name_
                }
    torch.save(checkpoint, filepath_)

if __name__ == '__main__':
    # Create DataLoaders for Training, Validation and Test
    train_loaders, valid_loaders, test_loaders = create_dataloader(data_dir)

    # create network
    model = create_network(model_name, hidden_layers)
    #print(model)

    # Get device to train on
    device = get_device()

    # Move model to device
    model = model.to(device)

    # Train the network
    train_network(model, epochs, learn_rate, train_loaders,
                valid_loaders, device, print_every_ = 5)

    print("Network training done ...")
    print("Saving checkpoint ...")
    # Save Checkpoint
    save_checkpoint(model_name, output_size, hidden_layers,
                    model, index_to_cat, cat_to_name, save_dir)
    print("Checkpoint saved at: {}".format(save_dir))
    print("### Script end ###")
