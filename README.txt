# Command to train:
- $ python3 train.py flowers --hidden_unit 1024 256 --arch RestNet101 --epochs 20 --save_dir 'RestNet101_script.pth' --gpu
- $ python3 train.py flowers --hidden_unit 1024 256 --arch RestNet50 --epochs 20 --save_dir 'RestNet50_script.pth' --gpu
- $ python3 train.py flowers --hidden_unit 4096 100 256 --arch VGG19 --epochs 30 --save_dir 'VGG19_script.pth' --gpu

# Command to test:
- $ python3 test.py flowers 'RestNet101_script.pth' --gpu
- $ python3 test.py flowers 'RestNet50_script.pth' --gpu
- $ python3 test.py flowers 'VGG19_script.pth' --gpu

# Command to predict
- $ python3 predict.py flowers/test/102/image_08030.jpg 'RestNet101_script.pth' --top_k 5 --gpu
- $ python3 predict.py flowers/test/102/image_08030.jpg 'RestNet50_script.pth' --top_k 5 --gpu
- $ python3 predict.py flowers/test/102/image_08030.jpg 'VGG19_script.pth' --top_k 5 --gpu

# Hardware
- GeForce GTX 1060, 6GB memory,  with CUDA 10.2

# VGG19
- Use about 4.7 GB of GPU memory while training
- Take 30 epochs to reach around 80% of validation accuracy
