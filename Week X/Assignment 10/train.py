import torch
import torch.utils.data as data
import numpy as np
import torch.nn as nn
from model import DenseNet
import os
import torch.optim as optim
import cv2 as cv

class DataSet(data.Dataset):
    def __init__(self,x,y,bboxes):
        self.x=np.load(x)
        self.y=np.load(y)
        self.bboxes=np.load(bboxes)

    def __getitem__(self, index):
        tmp=cv.cvtColor(self.x[index,:].reshape(64,64),cv.COLOR_GRAY2BGR)
        img=torch.from_numpy(tmp.astype(np.float32)/255.).permute(2,0,1)
        label=torch.from_numpy(self.y[index,:].astype(np.int32)).long()
        bbox=torch.from_numpy(self.bboxes[index,:].astype(np.float32))
        return img,label,bbox

    def __len__(self):
        return self.x.shape[0]

def train(epoch,model,opt,train_loader):
    global device,criterion_classification,criterion_box
    model.train()
    for i,(x,y,box) in enumerate(train_loader):
        x=x.to(device) #Nx1x64x64
        y=y.to(device) #Nx2
        box=box.to(device) #Nx4

        logit=model(x)#Nx28
        # print(logit.shape,y.shape)
        loss_class=criterion_classification(logit[:,:20].contiguous().view(-1,10),y.contiguous().view(-1))
        # print(logit[:,20:].shape,box.shape)
        loss_box=criterion_box(logit[:,20:],box.view(-1,8))
        loss=loss_class+loss_box*0.01
        
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 0.4)
        opt.step()

        if i%100:
            print("Epoch: [%d] [%d], loss class %.5f, loss box %.5f"%(epoch,i,loss_class.item(),
                                                                     loss_box.item()))

def test(epoch,model,test_loader):
    global device
    model.eval()
    correct=0
    loss_box_total=0
    for i, (x, y, box) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        box = box.to(device)

        #logit Nx28
        logit = model(x)
        logit_class=logit[:,:20].contiguous().view(-1,10)
        logit_box=logit[:,20:]

        loss_box=criterion_box(logit[:,20:],box.view(-1,8)).item()
        loss_box_total+=loss_box

        pred_class=logit_class.argmax(1).view(-1).cpu().numpy()
        target_class=y.view(-1).cpu().numpy()
        correct+=np.sum(pred_class==target_class)

    acc=correct/len(test_loader)/2
    print("Test Acc:",acc)
    print("loss box:",loss_box_total/len(test_loader))
    
    save_model(model,os.path.join(save_path,"epcoch_%d_acc%.4f_box%.4f.pth"%(epoch,acc,loss_box_total/len(test_loader))))
    

def save_model(model,filename):
    torch.save(model.state_dict(), filename.replace(".pth", '_params.pth'))
    torch.save(model, filename)

def update_learning_rate(opt,decay_rate=0.8,min_value=1e-3):
    for pg in opt.param_groups:
        pg["lr"]=max(pg["lr"]*decay_rate,min_value)
    print("learning rate",pg["lr"])

if __name__=="__main__":
    save_path="./result"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #############################
    #train dataloader
    train_dataset=DataSet("./data/train_X.npy",
                          "./data/train_Y.npy",
                          "./data/train_bboxes.npy",)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True,
                                               num_workers=1)
    #val dataloader
    val_dataset=DataSet("./data/valid_X.npy",
                          "./data/valid_Y.npy",
                          "./data/valid_bboxes.npy",)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True,
                                               num_workers=1)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # model=EfficientNet.from_pretrained( "efficientnet-b0", num_classes=28).to(device)
    model=DenseNet(growth_rate=32, block_config=(3, 6, 9, 12),
                 num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=28,in_channels=3).to(device)
    optimizer=optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())),lr=0.001,weight_decay=1e-4)

    criterion_classification=nn.CrossEntropyLoss().to(device)
    criterion_box= nn.MSELoss().to(device)
    
    for epoch in range(100):
        test(epoch,model,val_loader)
        train(epoch,model,optimizer,train_loader)
        
        update_learning_rate(optimizer,decay_rate=0.9,min_value=1e-3)
        
        
        
    

