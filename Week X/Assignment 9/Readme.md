# Assignment 9 Student# 1370023

## Instruction on I trained the model from scratch
NOTICE: Optional Batch Testing with argument doesn't work for me for some reason. 
So I manmually modified the parameters for training.

## MNIST
### Training flow with parameters
python3 A9_main.py dataset=0 mnist.c0=1 mnist.lr=0.001 mnist.weight_decay=0.0005 mnist.load_weights=0 mnist.n_epochs=20
### The commandline with arguments above doesn't work for me
dataset=0  c0=1  lr=0.0008  wd=0.0008  load_weight=2  n_epochs+=20
dataset=0  c0=1  lr=0.0005  wd=0.001  load_weight=2  n_epochs+=20
dataset=0  c0=1  lr=0.0003  wd=0.003  load_weight=2  n_epochs+=20


## FMNIST
### From scratch
### Training flow with parameters
python3 A9_main.py dataset=1 fmnist.c0=1 fmnist.lr=0.001 fmnist.weight_decay=0.0005 fmnist.load_weights=0 fmnist.n_epochs=20
### The commandline with arguments above doesn't work for me
dataset=1  c0=1  lr=0.0008  wd=0.0008  load_weight=2  n_epochs+=20
dataset=1  c0=1  lr=0.0005  wd=0.001  load_weight=2  n_epochs+=20
dataset=1  c0=1  lr=0.0003  wd=0.003  load_weight=2  n_epochs+=20


## Testing

## MNIST
python3 A9_main.py dataset=0 mnist.load_weights=1
## FMNIST
python3 A9_main.py dataset=1 fmnist.load_weight=1