import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class TrainParams:
    """
    :ivar optim_type: 0: SGD, 1: ADAM

    :ivar load_weights:
        0: train from scratch,
        1: load and test
        2: load if it exists and continue training

    :ivar save_criterion:  when to save a new checkpoint
        0: max validation accuracy
        1: min validation loss,
        2: max training accuracy
        3: min training loss

    :ivar vis: visualize the input and reconstructed images during validation and testing
    """

    def __init__(self):
        self.batch_size = 128
        self.optim_type = 0
        self.lr = 0.001
        self.momentum = 0.9
        self.n_epochs = 20
        self.weight_decay = 0.0005
        self.c0 = 1
        self.save_criterion = 0
        self.load_weights = 1
        self.weights_path = './checkpoints/model.pt'
        self.vis = 0


class MNISTParams(TrainParams):
    def __init__(self):
        super(MNISTParams, self).__init__()
        self.weights_path = './checkpoints/mnist/model.pt'


class FMNISTParams(TrainParams):
    def __init__(self):
        super(FMNISTParams, self).__init__()
        self.weights_path = './checkpoints/fmnist/model.pt'

""" I used the first method for finding optimal c0 which is by changing the parameter c0 to 0.8
    Therefore there is no need for implementation for the CompositeLoss class """
class CompositeLoss(nn.Module):
    def __init__(self, device):
        super(CompositeLoss, self).__init__()
        pass

    def init_weights(self):
        pass

    def forward(self, reconstruction_loiss, classification_loss):
        pass


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        # self.b1 = nn.Parameter(torch.zeros(520, requires_grad = True)).to(device)
        # self.b2 = nn.Parameter(torch.zeros(280, requires_grad = True)).to(device)
        # self.b3 = nn.Parameter(torch.zeros(140, requires_grad = True)).to(device)
        self.b1 = nn.Parameter(torch.zeros(520, requires_grad = True))
        self.b2 = nn.Parameter(torch.zeros(280, requires_grad = True))
        self.b3 = nn.Parameter(torch.zeros(140, requires_grad = True))

        # initialize the biasses
        # nn.init.constant_(self.b1, 0.0)
        # nn.init.constant_(self.b2, 0.0)
        # nn.init.constant_(self.b3, 0.0)

    def get_weights(self):
        return [self.w1, self.w2, self.w3]

    def init_weights(self):
        # self.w1 = nn.Parameter(torch.empty((520,28*28), requires_grad = True, dtype = torch.float)).to(device)
        # self.w2 = nn.Parameter(torch.empty((280,520), requires_grad = True, dtype = torch.float)).to(device)
        # self.w3 = nn.Parameter(torch.empty((140,280), requires_grad = True, dtype = torch.float)).to(device)
        self.w1 = nn.Parameter(torch.empty((520,28*28), requires_grad = True, dtype = torch.float))
        self.w2 = nn.Parameter(torch.empty((280,520), requires_grad = True, dtype = torch.float))
        self.w3 = nn.Parameter(torch.empty((140,280), requires_grad = True, dtype = torch.float))
        
        # xavier initialization for the weights
        torch.nn.init.xavier_normal_(self.w1)
        torch.nn.init.xavier_normal_(self.w2)
        torch.nn.init.xavier_normal_(self.w3)

    def forward(self, enc_input):
        x = enc_input.view(enc_input.size(0), -1)
        # f(x) = Wx + b notice: transpose is for matching the size
        x1 = F.relu(torch.mm(x, self.w1.transpose(0, 1)) + self.b1)
        x2 = F.relu(torch.mm(x1, self.w2.transpose(0, 1)) + self.b2)
        x3 = F.relu(torch.mm(x2, self.w3.transpose(0, 1)) + self.b3)
    
        return x3


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        # self.b4 = nn.Parameter(torch.zeros(280, requires_grad = True)).to(device)
        # self.b5 = nn.Parameter(torch.zeros(520, requires_grad = True)).to(device)
        # self.b6 = nn.Parameter(torch.zeros(28*28, requires_grad = True)).to(device)
        self.b4 = nn.Parameter(torch.zeros(280, requires_grad = True))
        self.b5 = nn.Parameter(torch.zeros(520, requires_grad = True))
        self.b6 = nn.Parameter(torch.zeros(28*28, requires_grad = True))

        # initialize the biasses
        # nn.init.constant_(self.b1, 0.0)
        # nn.init.constant_(self.b2, 0.0)
        # nn.init.constant_(self.b3, 0.0)

    def init_weights(self, shared_weights):
        # Share the weights with the Encoder Layer
        self.w4 = shared_weights[2]
        self.w5 = shared_weights[1]
        self.w6 = shared_weights[0]

    def forward(self, dec_input):
        x = dec_input.view(dec_input.size(0), -1)
        # f'(x) = WTx + b'
        x1 = F.relu(torch.mm(x, self.w4) + self.b4)
        x2 = F.relu(torch.mm(x1, self.w5) + self.b5)
        x3 = F.relu(torch.mm(x2, self.w6) + self.b6)
        x4 = x3.view(x.size(0), 1, 28, 28)

        return x4

# We ca simply use nn.Linear for the Classifier
class Classifier(nn.Module):
    def __init__(self, device):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(140, 80, True)
        self.fc2 = torch.nn.Linear(80, 40, True)
        self.fc3 = torch.nn.Linear(40, 10, True)

    def init_weights(self):
        pass

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return F.log_softmax(x3)
