# ImageClassifier-flower
This project aim to develope an images classifer to recognize different species of flowers. The dataset came fom [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) consisting of 102 flower commonly seen in the United Kingdom.

This project employ transfer-learning technique, so only the classifier layer of the original (pre-trained) network needed to be modify for this tasks.

### Dataset
Organized dataset, according to `torchvision` guidelines, can be download from [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation and testing.

### Pre-trained network
A few pre-trained networks has been tested including VGG-19, RestNet-50 and RestNet-101.

### Jupyter-notebook and Python script
This project divided in to two-stages.
- First stage, develop a network and train/validation using Jupyter-notebook so a results can be display easily.
- After done with hyper-pararmeters tunning, the Python script version is then develop for future deployment.

### Python script input argument
There are 3 Python script here
- `train.py`: For training a network where you can specify some hyper-parameters such as hidden-layers, archetecture of pre-trained network, number of epochs, bath-size, file-name of trained network checkpoint to be save, and flag if you wanted to train using GPU.
This training step will output epoch number, training-loss, validation-loss and accuracy on validation-dataset during training.
  - `python3 train.py flowers --hidden_unit 1024 256 --arch RestNet101 --epochs 20 --save_dir 'RestNet101_script.pth' --gpu`
  - `python3 train.py flowers --hidden_unit 1024 256 --arch RestNet50 --epochs 20 --save_dir 'RestNet50_script.pth' --gpu`
  - `python3 train.py flowers --hidden_unit 4096 100 256 --arch VGG19 --epochs 30 --save_dir 'VGG19_script.pth' --gpu`
- `validation.py`: The main purpose of this script is to test the checkpoint file saved from the training step, to make sure that the required information of the model has been saved and able to rebuild the network back again for later use.
  - `python3 test.py flowers 'RestNet101_script.pth' --gpu`
  - `python3 test.py flowers 'RestNet50_script.pth' --gpu`
  - `python3 test.py flowers 'VGG19_script.pth' --gpu`

- `test.py`: After the network has been trained, its a good practice to test the accuracy of the network with a dataset that its never work on before. This script take an image filename as an input along with checkpoint filename and number of predicted class to be report.
  - `python3 predict.py flowers/test/102/image_08030.jpg 'RestNet101_script.pth' --top_k 5 --gpu`
  - `python3 predict.py flowers/test/102/image_08030.jpg 'RestNet50_script.pth' --top_k 5 --gpu`
  - `python3 predict.py flowers/test/102/image_08030.jpg 'VGG19_script.pth' --top_k 5 --gpu`
