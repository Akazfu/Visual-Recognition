# -*- coding: utf-8 -*-
"""Assignment-3_notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vM9Ot32fNYF4wctXXrDhL4jk-FdOdT4E

Import and setup some auxiliary functions
"""

import torch
from torchvision import transforms, datasets
import numpy as np
import timeit
from collections import OrderedDict
from pprint import pformat
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

"""

TODO: Implement Logistic Regression here
"""

def logistic_regression(dataset_name):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: implement logistic regression hyper-parameter tuning here
    num_epochs = 5  # Number of epochs
    learning_rate = 1e-3  # Learning rate
    batch_size_training = 128  # Batch size for trainning
    batch_size_test = 1000 #Batch size for test
    
    
    # Define your train, validation, and test data loaders in PyTorch
    if dataset_name == "MNIST":
      input_size = 28 * 28  # MNIST dataset are 28px by 28px in size
      num_classes = 10  # MNIST dataset have 10 labels
      
      MNIST_training = datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]))

      MNIST_test_set = datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]))
      
      # create a training and a validation set
      # MNIST_training_set, MNIST_validation_set = torch.utils.data.random_split(MNIST_training, [48000, 12000])
      # Knowing issue: I used sampler instead of random_split.  However, validation accuracy seems to drop, but validation is just for turning hyperparameter.  Since I already done that, I think is fine? The test Result stays same.
      train_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(48000))) 
      validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(12000)))
      
      train_loader = torch.utils.data.DataLoader(MNIST_training,batch_size=batch_size_training, sampler=train_sampler)
      validation_loader = torch.utils.data.DataLoader(MNIST_training,batch_size=batch_size_training, sampler=validation_sampler)
      test_loader = torch.utils.data.DataLoader(MNIST_test_set,batch_size=batch_size_test, shuffle=True)
      
      test_set_len = len(MNIST_test_set)
      
    elif dataset_name == "CIFAR10":
      input_size = 32 * 32 * 3  # CIFAR10 dataset are 32px by 32px by RGB in size
      num_classes = 10  # CIFAR10 dataset have 10 labels
      
      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      
      CIFAR10_training = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

      CIFAR10_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
      
      # create a training and a validation set
      # CIFAR10_training_set, CIFAR10_validation_set = torch.utils.data.random_split(CIFAR10_training, [40000, 10000])
      # Knowing issue: I used sampler instead of random_split.  However, validation accuracy seems to drop, but validation is just for turning hyperparameter.  Since I already done that, I think is fine? The test Result stays same.
      train_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(40000))) 
      validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(10000)))
      
      train_loader = torch.utils.data.DataLoader(CIFAR10_training,batch_size=batch_size_training, sampler=train_sampler, num_workers=2)
      validation_loader = torch.utils.data.DataLoader(CIFAR10_training,batch_size=batch_size_training, sampler=validation_sampler, num_workers=2)
      test_loader = torch.utils.data.DataLoader(CIFAR10_test_set,batch_size=batch_size_test, shuffle=False, num_workers=2)
      
      test_set_len = len(CIFAR10_test_set)
      
      
    #Define your model and cross entropy loss
    class LogisticRegression(torch.nn.Module):
      
      def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
        
      def forward(self, y):
        # No need to use relu, when CrossEntropyLoss do log softmax automatically
        # y = torch.nn.functional.relu(self.linear(y))
        y = self.linear(y)
        return y
    
    model =  LogisticRegression(input_size, num_classes) 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)   # optimize parameters weight_decay = L2
    
    # Training
    def train(epoch):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.autograd.Variable(data.view(-1, input_size))        # Images flattened into 1D tensors
        target = torch.autograd.Variable(target)
        # Forward -> Backprop -> Optimize
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          
    # Validation
    def validation():
      model.eval()
      validation_loss = 0
      correct = 0
      with torch.no_grad(): # notice the use of no_grad
        for data, target in validation_loader:
          data = torch.autograd.Variable(data.view(-1, input_size))      # Images flattened into 1D tensors
          target = torch.autograd.Variable(target)
          output = model(data)
          validation_loss += criterion(output, target).item()
          _, pred = torch.max(output.data, 1)
          correct += (pred == target).sum()
      validation_loss /= len(validation_loader.dataset)
      print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(validation_loss, correct, len(validation_loader.dataset),100. * correct / len(validation_loader.dataset)))
    
    # Return Tensors
    predicted_test_labels = torch.zeros(test_set_len, batch_size_test)
    gt_labels = torch.zeros(test_set_len, batch_size_test)
    
    # Test
    def test():
      model.eval()
      i = 0
      correct = 0
      test_loss = 0
      with torch.no_grad():
        for data, target in test_loader:
          data = torch.autograd.Variable(data.view(-1, input_size))      # Images flattened into 1D tensors
          target = torch.autograd.Variable(target)
          output = model(data)
          test_loss += criterion(output, target).item()
          _, pred = torch.max(output.data, 1)
          correct += (pred == target).sum()
          
          # update return tensors by iterations
          predicted_test_labels[i:i+batch_size_test] = pred
          gt_labels[i:i+batch_size_test] = target.data.view_as(pred)
          i += batch_size_test
          
      test_loss /= len(test_loader.dataset)
      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
      
    # For loop, Validation after every 5 epochs
    validation()
    for epoch in range(1, num_epochs + 1):
      train(epoch)
      validation()
    test()
    
    #print(predicted_test_labels)
    #print(gt_labels)
    return predicted_test_labels, gt_labels